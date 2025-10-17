import json
from typing import Optional
import networkx as nx
import numpy as np
from matchms import Scores
from matchms.networking.networking_functions import get_top_hits
import pandas as pd


class SimilarityNetworkMod:
    """Create a spectral network from spectrum similarities."""

    def __init__(
        self,
        identifier_key: str = "spectrum_id",
        top_n: int = 20,
        max_links: int = 10,
        score_cutoff: float = 0.7,
        min_peaks: Optional[int] = None,  # allow None
        link_method: str = "single",
        keep_unconnected_nodes: bool = True,
    ):
        self.identifier_key = identifier_key
        self.top_n = top_n
        self.max_links = max_links
        self.score_cutoff = score_cutoff
        self.min_peaks = min_peaks
        self.link_method = link_method
        self.keep_unconnected_nodes = keep_unconnected_nodes
        self.graph: Optional[nx.Graph] = None

    def create_network(self, scores: Scores, score_name: str = None):
        if score_name is None:
            score_name = scores.scores.guess_score_name()
        assert self.top_n >= self.max_links, "top_n must be >= max_links"

        if scores.queries.shape != scores.references.shape:
            raise TypeError("Expected symmetric scores")
        if not np.all(scores.queries == scores.references):
            raise ValueError("Queries and references do not match")

        unique_ids = list({s.get(self.identifier_key) for s in scores.queries})

        # Initialize network graph
        msnet = nx.Graph()
        msnet.add_nodes_from(unique_ids)

        # Collect location and score of highest scoring candidates
        similars_idx, similars_scores = get_top_hits(
            scores,
            identifier_key=self.identifier_key,
            top_n=self.top_n,
            search_by="queries",
            score_name=score_name,
            ignore_diagonal=True,
        )

        # Build peaks_dict safely for any score type
        peaks_dict = {}
        for ref, query, scores_tuple in scores:
            if isinstance(scores_tuple, (float, np.float64)):
                n_matches = 1
            elif (
                isinstance(scores_tuple, (list, np.ndarray)) and len(scores_tuple) == 1
            ):
                n_matches = 1
            else:
                n_matches = scores_tuple[1]

            if ref.get(self.identifier_key) != query.get(self.identifier_key):
                peaks_dict[
                    (ref.get(self.identifier_key), query.get(self.identifier_key))
                ] = n_matches

        # Add edges
        for i, spec in enumerate(scores.queries):
            query_id = spec.get(self.identifier_key)
            ref_candidates = np.array(
                [
                    scores.references[x].get(self.identifier_key)
                    for x in similars_idx[query_id]
                ]
            )

            score_mask = similars_scores[query_id] >= self.score_cutoff
            peaks_mask = np.array(
                [
                    True
                    if self.min_peaks is None
                    else peaks_dict.get((ref_id, query_id), 0) >= self.min_peaks
                    for ref_id in ref_candidates
                ],
                dtype=bool,
            )
            self_link_mask = ref_candidates != query_id
            combined_mask = score_mask & peaks_mask & self_link_mask
            idx_all = np.where(combined_mask)[0]
            idx = idx_all[: self.max_links]

            if self.link_method == "single":
                new_edges = [
                    (
                        query_id,
                        str(ref_candidates[x]),
                        float(similars_scores[query_id][x]),
                    )
                    for x in idx
                ]
            elif self.link_method == "mutual":
                new_edges = [
                    (
                        query_id,
                        str(ref_candidates[x]),
                        float(similars_scores[query_id][x]),
                    )
                    for x in idx
                    if i in similars_idx[ref_candidates[x]][:]
                ]
            else:
                raise ValueError("Link method not known")

            msnet.add_weighted_edges_from(new_edges)

        if not self.keep_unconnected_nodes:
            msnet.remove_nodes_from(list(nx.isolates(msnet)))

        # Assign component IDs
        for i, comp in enumerate(nx.connected_components(msnet)):
            for node in comp:
                msnet.nodes[node]["component"] = i

        self.graph = msnet

    def filter_components(
        self, max_component_size: int = 0, cosine_delta: float = 0.02
    ):
        """Filter graph components by pruning edges if component size exceeds max_component_size."""
        if self.graph is None:
            raise ValueError("Graph not yet created. Run create_network first.")

        if max_component_size == 0:
            return

        big_components_present = True
        while big_components_present:
            big_components_present = False
            for component in list(nx.connected_components(self.graph)):
                if len(component) > max_component_size:
                    self._prune_component(component, cosine_delta)
                    big_components_present = True

        # Update component IDs
        for i, comp in enumerate(nx.connected_components(self.graph)):
            for node in comp:
                self.graph.nodes[node]["component"] = i

    def _prune_component(self, component, cosine_delta=0.02):
        """Remove weakest edges in a component until size is under threshold."""
        edges = self._get_edges_of_component(component)
        if not edges:
            return

        min_weight = min(edge[2]["weight"] for edge in edges)
        threshold = min_weight + cosine_delta

        for u, v, data in edges:
            if data["weight"] < threshold:
                self.graph.remove_edge(u, v)

    def _get_edges_of_component(self, component):
        """Return all unique edges belonging to a component."""
        edges = []
        seen = set()

        for node in component:
            for neighbor, data in self.graph[node].items():
                if neighbor in component:
                    edge_nodes = tuple(sorted([node, neighbor]))
                    if edge_nodes not in seen:
                        seen.add(edge_nodes)
                        edges.append((node, neighbor, data))
        return edges

    def min_component_size(self, min_size=None):
        """Remove components smaller than min_size."""
        if self.graph is None:
            raise ValueError("Graph not yet created. Run create_network first.")
        if min_size is None:
            return

        for component in list(nx.connected_components(self.graph)):
            if len(component) < min_size:
                for node in component:
                    self.graph.remove_node(node)

    def export_to_file(self, filename: str, graph_format: str = "graphml"):
        """
        Save the network to a file with chosen format.

        Parameters
        ----------
        filename
            Path to file to write to.
        graph_format
            Format, in which to store the network. Supported formats are: "cyjs", "gexf", "gml", "graphml", "json".
            Default is "graphml".
        """
        if not self.graph:
            raise ValueError(
                "No network found. Make sure to first run .create_network() step"
            )

        writer = self._generate_writer(graph_format)
        writer(filename)

    def _generate_writer(self, graph_format: str):
        writer = {
            "cyjs": self._export_to_cyjs,
            "gexf": self._export_to_gexf,
            "gml": self._export_to_gml,
            "graphml": self.export_to_graphml,
            "json": self._export_to_node_link_json,
        }

        assert graph_format in writer, (
            "Format not supported.\n"
            "Please use one of supported formats: 'cyjs', 'gexf', 'gml', 'graphml', 'json'"
        )
        return writer[graph_format]

    def export_to_graphml(self, filename: str):
        """Save the network as .graphml file.

        Parameters
        ----------
        filename
            Specify filename for exporting the graph.

        """
        nx.write_graphml_lxml(self.graph, filename)

    def _export_to_cyjs(self, filename: str):
        """Save the network in cyjs format."""
        graph = nx.cytoscape_data(self.graph)
        return self._write_to_json(graph, filename)

    def _export_to_node_link_json(self, filename: str):
        """Save the network in node-link format."""
        graph = nx.node_link_data(self.graph, edges="links")
        return self._write_to_json(graph, filename)

    @staticmethod
    def _write_to_json(graph: dict, filename: str):
        """Save the network as JSON file.

        Parameters
        ----------
        graph
            JSON-dictionary type graph to save.
        filename
            Specify filename for exporting the graph.

        """
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(graph, file)

    def _export_to_gexf(self, filename: str):
        """Save the network as .gexf file."""
        nx.write_gexf(self.graph, filename)

    def _export_to_gml(self, filename: str):
        """Save the network as .gml file."""
        nx.write_gml(self.graph, filename)

    def to_dataframe(self, col_name: str) -> pd.DataFrame:
        """Return DataFrame with specified attributes from the graph."""
        if self.graph is None:
            raise ValueError("Graph not yet created. Run create_network first.")

        data = []
        for node, attrs in self.graph.nodes(
            data=True,
        ):
            scan = node  # or: attrs.get('spectrum_id') or another identifier
            component = attrs.get("component", None)
            data.append({col_name: scan, "component": component})

        return pd.DataFrame(data)
