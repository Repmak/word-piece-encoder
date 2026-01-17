#pragma once
#include <vector>
#include <TokenizerInterface.h>

namespace sentencpp::embedding_utils {

    class VectorMaths {
        public:
            //
            static std::vector<float> mean_pooling(
                const std::vector<std::vector<float>>& token_embeddings,
                const std::vector<tokenizer::Token>& original_tokens
            );

            // Calculates the cosine similarity between two vectors.
            static float cosine_similarity(
                const std::vector<float>& vec_a,
                const std::vector<float>& vec_b
            );
    };

} // namespace sentencpp::embedding_utils
