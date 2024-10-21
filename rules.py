from const import ON, OFF


def _apply_rule(grid, i, j, num_neighbors: int, birth_map, death_map):
    if grid[i, j] == OFF:
        return birth_map[num_neighbors]

    return death_map[num_neighbors]


def conway(grid, i, j, num_neighbors: int):
    """Conway's Game of Life.

    Rules:
    - dead cells:
        - 3 neighbors -> cell comes alive
        - else -> cell stays dead
    - live cells:
        - 2 or 3 neighbors -> cell stays alive
        - else -> cell dies
    """
    birth_map = [OFF, OFF, OFF, ON, OFF, OFF, OFF, OFF, OFF]
    death_map = [OFF, OFF, ON, ON, OFF, OFF, OFF, OFF, OFF]

    return _apply_rule(grid, i, j, num_neighbors, birth_map, death_map)


def _conway(i, j, total, grid, newGrid, activity):
    if grid[i, j] == ON:
        if (total < 2) or (total > 3):
            newGrid[i, j] = OFF
            # change occurred
            activity[i, j] += 1
    else:
        if total == 3:
            newGrid[i, j] = ON
            activity[i, j] += 1
            # change occurred


def fig9(i, j, total, grid, newGrid, activity):
    death = [0, 1, 7, 8]
    if grid[i, j] == ON:
        if total in death:
            newGrid[i, j] = OFF
            activity[i, j] += 1
    else:
        if total == 4:
            newGrid[i, j] = ON
            activity[i, j] += 1


def fig6(grid, i, j, num_neighbors: int):
    birth_map = [OFF, OFF, OFF, ON, OFF, OFF, ON, OFF, OFF]  # if dead
    death_map = [OFF, OFF, ON, ON, OFF, ON, ON, OFF, OFF]  # if alive

    if grid[i, j] == OFF:
        return birth_map[num_neighbors]
    return death_map[num_neighbors]


def _fig6(i, j, total, grid, newGrid, activity):
    death = [0, 1, 4, 7, 8]
    birth = [3, 6]
    if grid[i, j] == ON:
        if total in death:
            newGrid[i, j] = OFF
            activity[i, j] += 1
    else:
        if total in birth:
            newGrid[i, j] = ON
            activity[i, j] += 1


def some(i, j, total, grid, newGrid, activity):
    death_map = [True, True, False, False, False, False, True, False, False]
    birth_map = [
        False,
        False,
        False,
        True,
        False,
        True,
        False,
        False,
        False,
    ]
    # death = [0, 1, 6, 7, 8]
    # birth = [3, 4]
    if grid[i, j] == ON:
        if death_map[total]:
            newGrid[i, j] = OFF
            activity[i, j] += 1
    else:
        if birth_map[total]:
            newGrid[i, j] = ON
            activity[i, j] += 1
