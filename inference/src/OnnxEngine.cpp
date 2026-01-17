#include <iostream>
#include <algorithm>
#include "OnnxEngine.h"


namespace sentencpp::inference {

    OnnxEngine::OnnxEngine(
        const std::string& model_path,
        const ModelConfig& config
    ) :
        config_(config),
        env(ORT_LOGGING_LEVEL_WARNING, "BERT_Inference"),
        session(nullptr),
        memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
    {
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // Load model.
        session = Ort::Session(env, model_path.c_str(), session_options);

        // Discover input and output names.
        Ort::AllocatorWithDefaultOptions allocator;
        for (size_t i = 0; i < session.GetInputCount(); i++) {
            input_names.emplace_back(session.GetInputNameAllocated(i, allocator).get());
        }
        for (size_t i = 0; i < session.GetOutputCount(); i++) {
            output_names.emplace_back(session.GetOutputNameAllocated(i, allocator).get());
        }
    }


    // PUBLIC METHODS --------------------------------------------------------------------------------------------------

    std::vector<std::vector<float>> OnnxEngine::encode(const std::vector<tokenizer::Token>& tokens) {
        const size_t sequence_length = tokens.size();
        if (sequence_length == 0) return {};

        // Extract data from each Token instance.
        std::vector<int64_t> input_ids, attention_mask, segment_ids;
        input_ids.reserve(sequence_length);
        attention_mask.reserve(sequence_length);
        segment_ids.reserve(sequence_length);

        for (const auto& t : tokens) {
            input_ids.push_back(t.id);
            attention_mask.push_back(t.attention_mask);
            segment_ids.push_back(t.segment_id);
        }

        std::vector<Ort::Value> input_tensors;
        std::vector<const char*> in_names;
        const std::vector<int64_t> input_shape = {1, static_cast<int64_t>(sequence_length)};

        for (const auto& name : input_names) {
            in_names.push_back(name.c_str());
            if (name == config_.input_ids_name) {
                input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
                    memory_info, input_ids.data(), input_ids.size(), input_shape.data(), input_shape.size()
                ));
            } else if (name == config_.attention_mask_name) {
                input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
                    memory_info, attention_mask.data(), attention_mask.size(), input_shape.data(), input_shape.size()
                ));
            } else if (name == config_.token_type_ids_name) {
                input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
                    memory_info, segment_ids.data(), segment_ids.size(), input_shape.data(), input_shape.size()
                ));
            }
        }

        std::vector<const char*> out_names;
        for (const auto& name : output_names) out_names.push_back(name.c_str());

        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            in_names.data(),
            input_tensors.data(),
            input_tensors.size(),
            out_names.data(),
            out_names.size()
        );

        // Find the output data line.
        size_t output_idx = -1;
        for (size_t i = 0; i < output_names.size(); ++i) {
            if (output_names[i] == config_.output_name) {
                output_idx = i;
                break;
            }
        }

        if (output_idx == -1) {
            std::cerr << "Unable to identify output line " << config_.output_name << std::endl;
            exit(-1);
        }

        // Parse Output.
        auto& output_tensor = output_tensors[output_idx];
        float* output_data = output_tensor.GetTensorMutableData<float>();
        const auto shape_info = output_tensor.GetTensorTypeAndShapeInfo().GetShape();

        std::vector<std::vector<float>> embeddings;

        if (shape_info.size() == 3) {
            // 3D output.
            const int64_t num_tokens = shape_info[1];
            const int64_t hidden_size = shape_info[2];

            embeddings.reserve(num_tokens);
            for (int64_t i = 0; i < num_tokens; ++i) {
                float* start = output_data + (i * hidden_size);
                embeddings.emplace_back(start, start + hidden_size);
            }
        }
        else if (shape_info.size() == 2) {
            // 2D output.
            const int64_t hidden_size = shape_info[1];
            embeddings.emplace_back(output_data, output_data + hidden_size);
        }

        return embeddings;
    }


    // PRIVATE METHODS -------------------------------------------------------------------------------------------------

    // void ORTWrapper::post_processing(std::vector<float>& raw_logits, size_t num_tokens) {}
    //
    // std::vector<float> ORTWrapper::perform_pooling(const std::vector<float>& raw_logits, size_t num_tokens) {
    //     const size_t embedding_dim = 384;  // Fixed for all-MiniLM-L6-v2.
    //     std::vector<float> sentence_embedding(embedding_dim, 0.0f);
    //
    //     for (size_t i = 0; i < num_tokens; ++i) {
    //         for (size_t d = 0; d < embedding_dim; ++d) {
    //             sentence_embedding[d] += raw_logits[i * embedding_dim + d];
    //         }
    //     }
    //
    //     // Divide by the number of tokens to get the average.
    //     for (float& val : sentence_embedding) val /= static_cast<float>(num_tokens);
    //
    //     return sentence_embedding;
    // }
    //
    // std::vector<float> ORTWrapper::calculate_softmax(const std::vector<float>& logits) {
    //     std::vector<float> probabilities(logits.size());
    //
    //     // Find max element.
    //     float max_logit = *std::max_element(logits.begin(), logits.end());
    //
    //     float sum = 0.0f;
    //     for (size_t i = 0; i < logits.size(); ++i) {
    //         probabilities[i] = std::exp(logits[i] - max_logit);
    //         sum += probabilities[i];
    //     }
    //
    //     for (float& p : probabilities) {
    //         p /= sum;
    //     }
    //
    //     return probabilities;
    // }

} // namespace sentencpp::inference
