class GisNode:
    """GIS Tree Node housing children pointers and contents"""

    def __init__(self, gis_join: str, contents: dict):
        self.gis_join = gis_join
        self.contents = contents
        self.children = {}

    def has_children(self):
        return len(self.children) > 0

    def is_root(self):
        return self.gis_join == ""

    def __repr__(self):
        return f"GisNode: gis_join={self.gis_join}, metadata={self.contents}, children={self.children.keys()}"


class GisTree:
    """Radix-like tree for housing hierarchical GISJOIN metadata"""

    def __init__(self):
        self.root: GisNode = GisNode("", {})
        self.state_prefix_length: int = 4
        self.county_prefix_length: int = self.state_prefix_length + 4

    def insert_county(self, gis_join: str, contents: dict) -> None:
        county_to_insert = GisNode(gis_join, contents)

        state_gis_join: str = gis_join[:self.state_prefix_length]
        if state_gis_join not in self.root.children:
            self.insert_state(state_gis_join)

        state: GisNode = self.root.children[state_gis_join]
        state.children[state_gis_join] = county_to_insert

    def insert_state(self, gis_join: str) -> None:
        self.root.children[gis_join] = GisNode(gis_join, {})

    def get_county(self, gis_join: str) -> GisNode:
        state: GisNode = self.get_state(gis_join)
        return state.children[gis_join]

    def get_counties(self, gis_join: str) -> list:  # returns list(GisNode)
        state_gis_join: str = gis_join[:self.state_prefix_length]
        return list(self.root.children[state_gis_join].children.values())

    def get_state(self, gis_join: str) -> GisNode:
        state_gis_join: str = gis_join[:self.state_prefix_length]
        return self.root.children[state_gis_join]

    def get_states(self) -> list:  # returns list(GisNode)
        return list(self.root.children.values())
