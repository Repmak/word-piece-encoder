#pragma once

#include <string>
#include <vector>
#include <sentenCPP/tokenizer/TokenizerInterface.h>

#include "InferenceInterface.h"


namespace sentencpp::inference {

    struct ModelConfig {
        std::string model_path;
        std::string input_ids_name = "input_ids";
        std::string attention_mask_name = "attention_mask";
        std::string token_type_ids_name = "token_type_ids";
        std::string output_name = "last_hidden_state";
    };

    class OnnxEngine : public InferenceInterface {
        public:
            explicit OnnxEngine(const ModelConfig& config);

            [[nodiscard]] std::vector<std::vector<float>> encode(const std::vector<tokenizer::Token>& tokens) override;

        private:
            ModelConfig config_;  // For configuring data lines in/out of the model.

            Ort::Env env;
            Ort::Session session;
            Ort::MemoryInfo memory_info;

            std::vector<std::string> input_names;
            std::vector<std::string> output_names;
    };

} // namespace sentencpp::inference
