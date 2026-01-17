#pragma once

#include <TokenizerInterface.h>
#include <onnxruntime_cxx_api.h>
#include <vector>

namespace sentencpp::inference {

    class InferenceInterface {
        public:
            virtual ~InferenceInterface() = default;

            // Encodes Token objects into their vector embeddings.
            [[nodiscard]] virtual std::vector<std::vector<float>> encode(const std::vector<tokenizer::Token>& tokens) = 0;

            // std::vector<float> run(const std::vector<tokenizer::Token>& tokens);
    };

} // namespace sentencpp::inference
