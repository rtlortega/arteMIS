class SimilarityCalculator:
    def __init__(self, filtered_spectra):
        self.filtered_spectra = filtered_spectra
        self.spec2vec_similarity = None
        self.ms2ds_model = None

    # --- Modified Cosine ---
    def calculate_modcosine(self, tolerance=0.01):
        from matchms import calculate_scores
        from matchms.similarity import ModifiedCosine

        modcosine_similarity = ModifiedCosine(tolerance=tolerance)
        scores = calculate_scores(
            references=self.filtered_spectra,
            queries=self.filtered_spectra,
            similarity_function=modcosine_similarity,
        )
        return scores

    # --- Spec2Vec ---
    def load_spec2vec(self, model_path):
        import gensim
        from spec2vec import Spec2Vec

        model_Spec2Vec = gensim.models.Word2Vec.load(model_path)
        self.spec2vec_similarity = Spec2Vec(
            model=model_Spec2Vec,
            intensity_weighting_power=0.5,
            allowed_missing_percentage=5.0,
        )

    def calculate_spec2vec(self):
        from matchms import calculate_scores

        if self.spec2vec_similarity is None:
            raise ValueError("Spec2Vec model not loaded. Call load_spec2vec() first.")
        scores = calculate_scores(
            references=self.filtered_spectra,
            queries=self.filtered_spectra,
            similarity_function=self.spec2vec_similarity,
        )
        return scores

    # --- MS2DeepScore ---
    def load_ms2deepscore(self, model_file):
        from ms2deepscore.models import load_model
        from ms2deepscore import MS2DeepScore

        model_MS2DS = load_model(model_file)
        # Wrap it in MS2DeepScore so it can be passed to calculate_scores
        self.ms2ds_similarity = MS2DeepScore(model=model_MS2DS)

    def calculate_ms2deepscore(self):
        from matchms import calculate_scores

        if self.ms2ds_similarity is None:
            raise ValueError(
                "MS2DeepScore model not loaded. Call load_ms2deepscore() first."
            )
        scores = calculate_scores(
            references=self.filtered_spectra,
            queries=self.filtered_spectra,
            similarity_function=self.ms2ds_similarity,
        )
        return scores
