from const import ON, OFF


class Rule:
    def __init__(self, birth_map, death_map) -> None:
        # lists where the index is a cells number of neighbors and
        # the value is the state of the cell in the next generation
        self._birth_map = birth_map # if dead
        self._death_map = death_map # if alive

    @property
    def birth_map(self):
        return self._birth_map

    @property
    def death_map(self):
        return self._death_map


"""Conway's Game of Life.

Rules:
- dead cells:
    - 3 neighbors -> cell comes alive
    - else -> cell stays dead
- live cells:
    - 2 or 3 neighbors -> cell stays alive
    - else -> cell dies
"""
conway = Rule(
    birth_map=[OFF, OFF, OFF, ON, OFF, OFF, OFF, OFF, OFF],
    death_map=[OFF, OFF, ON, ON, OFF, OFF, OFF, OFF, OFF]
)


fig6 = Rule(
    birth_map=[OFF, OFF, OFF, ON, OFF, OFF, ON, OFF, OFF],
    death_map=[OFF, OFF, ON, ON, OFF, ON, ON, OFF, OFF]
)


fig8 = Rule(
    birth_map=[OFF, OFF, OFF, OFF, ON, OFF, OFF, OFF, OFF],
    death_map=[OFF, OFF, ON, ON, ON, ON, ON, OFF, OFF]
)