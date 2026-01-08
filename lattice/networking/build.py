from .SimilarityNetworkMod import SimilarityNetworkMod


def build_similarity_graph(
    scores,
    score_name,
    identifier_key,
    cut_off,
    max_links,
    max_comp_size,
    link_method="single",
    min_peaks=None,
    top_n=50,
):
    """
    Build a filtered similarity graph from a matchms Scores object.
    Returns a networkx graph (copy).
    Adds some of the GNPS functionalities
    """
    net = SimilarityNetworkMod(
        identifier_key=identifier_key,
        score_cutoff=cut_off,
        max_links=max_links,
        min_peaks=min_peaks,
        top_n=50,
        link_method=link_method,
    )
    net.create_network(scores, score_name=score_name)
    net.filter_components(max_component_size=max_comp_size)
    return net.graph.copy()
