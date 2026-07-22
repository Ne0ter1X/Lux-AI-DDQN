import numpy as np
from lux.game import Game


def get_inputs(game_state):
    w, h = game_state.map.width, game_state.map.height
    M = [
        [0 if game_state.map.map[j][i].resource == None else game_state.map.map[j][i].resource.amount for i in range(w)]
        for j in range(h)]
    M = np.array(M).reshape((h, w, 1))

    U_player = [[[0, 0, 0, 0, 0] for i in range(w)] for j in range(h)]
    player = game_state.players[game_state.id]
    units = player.units
    for i in units:
        U_player[i.pos.y][i.pos.x] = [i.type, i.cooldown, i.cargo.wood, i.cargo.coal, i.cargo.uranium]
    U_player = np.array(U_player)

    U_opponent = [[[0, 0, 0, 0, 0] for i in range(w)] for j in range(h)]
    opponent = game_state.players[(game_state.id + 1) % 2]
    units = opponent.units
    for i in units:
        U_opponent[i.pos.y][i.pos.x] = [i.type, i.cooldown, i.cargo.wood, i.cargo.coal, i.cargo.uranium]
    U_opponent = np.array(U_opponent)

    player = game_state.players[game_state.id]
    e = player.cities
    C_player = [[[0, 0, 0] for i in range(w)] for j in range(h)]
    for k in e:
        citytiles = e[k].citytiles
        for i in citytiles:
            C_player[i.pos.y][i.pos.x] = [i.cooldown, e[k].fuel, e[k].light_upkeep]
    C_player = np.array(C_player)

    opponent = game_state.players[(game_state.id + 1) % 2]
    e = opponent.cities
    C_opponent = [[[0, 0, 0] for i in range(w)] for j in range(h)]
    for k in e:
        citytiles = e[k].citytiles
        for i in citytiles:
            C_opponent[i.pos.y][i.pos.x] = [i.cooldown, e[k].fuel, e[k].light_upkeep]
    C_opponent = np.array(C_opponent)

    turn = game_state.turn
    day_phase = np.full((h, w, 1), (turn % 40) / 39.0)
    night_flag = np.full((h, w, 1), 1.0 if (turn % 40) >= 30 else 0.0)
    game_progress = np.full((h, w, 1), turn / 359.0)

    E = np.dstack([M, U_opponent, U_player, C_opponent, C_player,
                   day_phase, night_flag, game_progress]).astype(np.float32)

    E[:, :, 0] = E[:, :, 0] / 800.0
    E[:, :, 2] = E[:, :, 2] / 6.0
    E[:, :, 7] = E[:, :, 7] / 6.0
    E[:, :, 3:6] = E[:, :, 3:6] / 100.0
    E[:, :, 8:11] = E[:, :, 8:11] / 100.0
    E[:, :, 11] = E[:, :, 11] / 10.0
    E[:, :, 14] = E[:, :, 14] / 10.0
    E[:, :, 12] = E[:, :, 12] / 2000.0
    E[:, :, 15] = E[:, :, 15] / 2000.0
    E[:, :, 13] = E[:, :, 13] / 200.0
    E[:, :, 16] = E[:, :, 16] / 200.0

    return E

def get_action_mask_from_state(game_state, observation_player):
    """
    Return a (H, W, 8) float32 mask using the live game_state object.
    Args:
        game_state: the current Game object
        observation_player: the player ID (0 or 1)
    """
    player_id = observation_player
    player = game_state.players[player_id]
    w, h = game_state.map.width, game_state.map.height

    mask = np.zeros((h, w, 8), dtype=np.float32)

    # pre-compute global counts needed for the worker-limit rule
    total_units = len(player.units)
    total_city_tiles = sum(len(c.citytiles) for c in player.cities.values())

    # ----- unit actions (0-5) -----
    for unit in player.units:
        y, x = unit.pos.y, unit.pos.x
        mask[y, x, 0] = 1.0
        if unit.cooldown < 1:
            # moves (0-4) are always legal for a ready unit
            mask[y, x, 0:5] = 1.0
            # build city (5) - only if cargo ≥ 100 and tile is buildable
            if unit.can_build(game_state.map):
                mask[y, x, 5] = 1.0

    # ----- city-tile actions (6-7) -----
    for city in player.cities.values():
        for ct in city.citytiles:
            y, x = ct.pos.y, ct.pos.x
            if ct.cooldown < 1:
                # research (6) - always available if not on cooldown
                mask[y, x, 6] = 1.0
                # build worker (7) - only if we haven't hit the cap
                if total_units < total_city_tiles:
                    mask[y, x, 7] = 1.0

    return mask


def get_prediction_actions(y, player, game_state, option):
    actions = []

    for unit in player.units:
        y, x = unit.pos.y, unit.pos.x
        a = option[y, x]
        if a == 0:  # center (do nothing / stay)
            actions.append(unit.move("c"))
        elif a == 1:  # south
            actions.append(unit.move("s"))
        elif a == 2:  # north
            actions.append(unit.move("n"))
        elif a == 3:  # west
            actions.append(unit.move("w"))
        elif a == 4:  # east
            actions.append(unit.move("e"))
        elif a == 5:  # build city
            # If the mask didn't prevent this, the unit can build.
            actions.append(unit.build_city())
        # else:
        #     # should not happen if mask is correct
        #     # actions.append(unit.move("c"))

    for city in player.cities.values():
        for ct in city.citytiles:
            y, x = ct.pos.y, ct.pos.x
            a = option[y, x]
            if a == 6:  # research
                actions.append(ct.research())
            elif a == 7:  # build worker
                actions.append(ct.build_worker())
            # else:
            #     actions.append(ct.research())  # safety default

    return actions

def calculate_shaped_reward(prev_gs, curr_gs, pid):
    if prev_gs is None:
        return 0.0
    p_prev, p_curr = prev_gs.players[pid], curr_gs.players[pid]
    o_prev, o_curr = prev_gs.players[(pid+1)%2], curr_gs.players[(pid+1)%2]

    reward = 0.0

    # city‑tiles (primary objective)
    n_city_prev = sum(len(c.citytiles) for c in p_prev.cities.values())
    n_city_curr = sum(len(c.citytiles) for c in p_curr.cities.values())
    reward += 0.02 * (n_city_curr - n_city_prev)

    # workers
    reward += 0.005 * (len(p_curr.units) - len(p_prev.units))

    # research points
    reward += 0.0002 * (p_curr.research_points - p_prev.research_points)

    # total city fuel
    fuel_prev = sum(c.fuel for c in p_prev.cities.values())
    fuel_curr = sum(c.fuel for c in p_curr.cities.values())
    reward += 0.0001 * (fuel_curr - fuel_prev)

    # resources carried by units (wood‑equivalent)
    def wood_equiv(u):
        return u.cargo.wood + 10*u.cargo.coal + 40*u.cargo.uranium
    res_prev = sum(wood_equiv(u) for u in p_prev.units)
    res_curr = sum(wood_equiv(u) for u in p_curr.units)
    reward += 0.00005 * (res_curr - res_prev)

    # opponent city‑tile delta
    o_city_prev = sum(len(c.citytiles) for c in o_prev.cities.values())
    o_city_curr = sum(len(c.citytiles) for c in o_curr.cities.values())
    reward -= 0.1 * (n_city_prev - n_city_curr)     # lost our own tiles (already counted above, but symmetric)

    # night survival bonus (small but continuous)
    if curr_gs.turn % 40 >= 30:                      # night turn
        reward += 0.005 * n_city_curr                   # each surviving city gets a pat on the back

    if not p_prev.researched_coal() and p_curr.researched_coal():
        reward += 0.003
    if not p_prev.researched_uranium() and p_curr.researched_uranium():
        reward += 0.005

    return reward

def get_final_outcome(steps, player_id):
    """
    Inspect the last step of an episode and return:
        +1   if player_id won
        -1   if player_id lost
         0   if tie or timeout (shouldn't happen normally)
    """
    last = steps[-1]                          # list of state dicts for each agent
    p_state = last[player_id]
    o_state = last[(player_id + 1) % 2]

    # If the player is DONE and has zero units and zero cities, they lost.
    if p_state['status'] in ('DONE', 'ERROR', 'INVALID'):
        p_cities = p_state['observation'].get('cities', {})
        p_units  = p_state['observation'].get('units', [])
        if len(p_cities) == 0 and len(p_units) == 0:
            return -1

    # If the opponent is DONE with zero cities and zero units, the player won.
    if o_state['status'] in ('DONE', 'ERROR', 'INVALID'):
        o_cities = o_state['observation'].get('cities', {})
        o_units  = o_state['observation'].get('units', [])
        if len(o_cities) == 0 and len(o_units) == 0:
            return 1

    # Both ended normally (step 359) – compare city‑tile counts.
    p_city_count = len(p_state['observation'].get('cities', {}))
    o_city_count = len(o_state['observation'].get('cities', {}))
    if p_city_count > o_city_count:
        return 1
    elif p_city_count < o_city_count:
        return -1
    return 0
