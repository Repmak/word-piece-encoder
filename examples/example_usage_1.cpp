#include <iostream>
#include <iomanip>

#include <sentenCPP/tokenizer/WordPiece.h>
#include <sentenCPP/inference/OnnxEngine.h>
#include <sentenCPP/embedding_utils/VectorMaths.h>


int main() {
    try {
        std::string project_root = SENTENCPP_SOURCE_DIR;
        std::string config_path = "/Users/jkamper/Documents/sentence-transformers-all-mini-lm-l6-v2/tokenizer.json";
        std::string model_path = "/Users/jkamper/Documents/sentence-transformers-all-mini-lm-l6-v2/model.onnx";

        sentencpp::tokenizer::WordPieceConfig config;
        config.config_path = config_path;
        config.vocab_key = "/model/vocab";
        const sentencpp::tokenizer::WordPiece tokenizer(config);

        // std::cout << tokenizer.get_vocab_list() << std::endl;

        const std::string text = "The weather is great!";
        const auto tokens = tokenizer.tokenize(text);
        const std::string text2 = "the weather is bad";
        const auto tokens2 = tokenizer.tokenize(text2);

        // for (size_t i = 0; i < tokens.size(); ++i) std::cout << i << ": " << tokens[i] << "\n";
        // for (size_t i = 0; i < tokens.size(); ++i) std::cout << i << ": " << tokens2[i] << "\n";

        sentencpp::inference::ModelConfig model_config;
        model_config.model_path = model_path;
        sentencpp::inference::OnnxEngine engine(model_config);

        // Note: It is normal for vector embeddings of padding tokens to not store absolute zeros. Attention mechanism will affect the values.
        std::vector<std::vector<float>> embeddings = engine.encode(tokens);
        std::vector<std::vector<float>> embeddings2 = engine.encode(tokens2);

        auto sent_vec_1 = sentencpp::embedding_utils::VectorMaths::mean_pooling(embeddings, tokens);
        auto sent_vec_2 = sentencpp::embedding_utils::VectorMaths::mean_pooling(embeddings2, tokens2);

        float similarity = sentencpp::embedding_utils::VectorMaths::cosine_similarity(sent_vec_1, sent_vec_2);
        std::cout << "Similarity Score: " << similarity << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}
