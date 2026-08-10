// Fashion MNIST inference: bare-metal 784→128→64→10 MLP.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>

constexpr int INPUT_SIZE  = 784;
constexpr int HIDDEN1     = 128;
constexpr int HIDDEN2     = 64;
constexpr int NUM_CLASSES = 10;
constexpr int NUM_TEST    = 1000;

struct Tensor {
    float* data = nullptr;
    int rows = 0;
    int cols = 0;
    bool owns_data = true;

    Tensor() = default;

    Tensor(int r, int c) : rows(r), cols(c) {
        data = new float[r * c]();
        owns_data = true;
    }

    Tensor(float* ptr, int r, int c) : data(ptr), rows(r), cols(c), owns_data(false) {}   // non-owning view

    ~Tensor() {
        if (owns_data && data) {
            delete[] data;
            data = nullptr;
        }
    }

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    Tensor(Tensor&& other) noexcept
        : data(other.data), rows(other.rows), cols(other.cols), owns_data(other.owns_data) {
        other.data = nullptr;
        other.rows = 0;
        other.cols = 0;
        other.owns_data = false;
    }

    Tensor& operator=(Tensor&& other) noexcept {
        if (this == &other) return *this;
        if (owns_data && data) delete[] data;
        data = other.data;
        rows = other.rows;
        cols = other.cols;
        owns_data = other.owns_data;
        other.data = nullptr;
        other.rows = 0;
        other.cols = 0;
        other.owns_data = false;
        return *this;
    }

    float& operator()(int r, int c) { return data[r * cols + c]; }
    const float& operator()(int r, int c) const { return data[r * cols + c]; }
};

struct Layer {
    Tensor weights;   // (in_features, out_features) row-major
    Tensor bias;      // (1, out_features)
    int in_features;
    int out_features;

    Layer() = default;

    Layer(int in_f, int out_f)
        : weights(in_f, out_f), bias(1, out_f), in_features(in_f), out_features(out_f) {}

    Tensor forward_relu(const Tensor& input) const {
        Tensor output(1, out_features);
        for (int j = 0; j < out_features; j++) {
            float sum = bias.data[j];
            for (int i = 0; i < in_features; i++) {
                sum += input.data[i] * weights.data[i * out_features + j];
            }
            output.data[j] = std::max(0.0f, sum);
        }
        return output;
    }

    Tensor forward_linear(const Tensor& input) const {
        Tensor output(1, out_features);
        for (int j = 0; j < out_features; j++) {
            float sum = bias.data[j];
            for (int i = 0; i < in_features; i++) {
                sum += input.data[i] * weights.data[i * out_features + j];
            }
            output.data[j] = sum;
        }
        return output;
    }
};

bool read_binary(const char* path, void* dst, size_t expected_bytes) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open file: %s\n", path);
        return false;
    }
    size_t read = fread(dst, 1, expected_bytes, f);
    fclose(f);
    if (read != expected_bytes) {
        fprintf(stderr, "ERROR: File %s: read %zu bytes, expected %zu\n", path, read, expected_bytes);
        return false;
    }
    return true;
}

bool load_weights(const char* path, Layer& layer1, Layer& layer2, Layer& layer3) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open weights file: %s\n", path);
        return false;
    }

    size_t n;

    // Python exports transposed (784, 128) row-major.
    n = fread(layer1.weights.data, sizeof(float), INPUT_SIZE * HIDDEN1, f);
    if (n != INPUT_SIZE * HIDDEN1) {
        fprintf(stderr, "ERROR: Layer1 weights: read %zu, expected %d\n", n, INPUT_SIZE * HIDDEN1);
        fclose(f);
        return false;
    }

    n = fread(layer1.bias.data, sizeof(float), HIDDEN1, f);
    if (n != HIDDEN1) {
        fprintf(stderr, "ERROR: Layer1 bias: read %zu, expected %d\n", n, HIDDEN1);
        fclose(f);
        return false;
    }

    n = fread(layer2.weights.data, sizeof(float), HIDDEN1 * HIDDEN2, f);
    if (n != HIDDEN1 * HIDDEN2) {
        fprintf(stderr, "ERROR: Layer2 weights: read %zu, expected %d\n", n, HIDDEN1 * HIDDEN2);
        fclose(f);
        return false;
    }
    n = fread(layer2.bias.data, sizeof(float), HIDDEN2, f);
    if (n != HIDDEN2) {
        fprintf(stderr, "ERROR: Layer2 bias: read %zu, expected %d\n", n, HIDDEN2);
        fclose(f);
        return false;
    }

    n = fread(layer3.weights.data, sizeof(float), HIDDEN2 * NUM_CLASSES, f);
    if (n != HIDDEN2 * NUM_CLASSES) {
        fprintf(stderr, "ERROR: Layer3 weights: read %zu, expected %d\n", n, HIDDEN2 * NUM_CLASSES);
        fclose(f);
        return false;
    }
    n = fread(layer3.bias.data, sizeof(float), NUM_CLASSES, f);
    if (n != NUM_CLASSES) {
        fprintf(stderr, "ERROR: Layer3 bias: read %zu, expected %d\n", n, NUM_CLASSES);
        fclose(f);
        return false;
    }

    fclose(f);
    return true;
}

void softmax(const float* input, float* output, int size) {
    float max_val = *std::max_element(input, input + size);
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }
    for (int i = 0; i < size; i++) {
        output[i] /= sum;
    }
}

const char* CLASS_NAMES[] = {
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
};

int main(int argc, char* argv[]) {
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("Fashion MNIST C++ Inference Engine (Standalone)\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");

    std::string base_dir = ".";
    if (argc > 0) {
        std::string exe_path = argv[0];
        size_t last_slash = exe_path.find_last_of("/\\");
        if (last_slash != std::string::npos) {
            base_dir = exe_path.substr(0, last_slash);
        }
    }

    std::string weights_path = base_dir + "/weights_fashion.bin";
    std::string images_path  = base_dir + "/test_images.bin";
    std::string labels_path  = base_dir + "/test_labels.bin";
    std::string results_path = base_dir + "/fashion_mnist_results.txt";

    printf("[1/4] Loading weights from %s...\n", weights_path.c_str());

    Layer layer1(INPUT_SIZE, HIDDEN1);
    Layer layer2(HIDDEN1, HIDDEN2);
    Layer layer3(HIDDEN2, NUM_CLASSES);

    if (!load_weights(weights_path.c_str(), layer1, layer2, layer3)) {
        return 1;
    }
    printf("  Layer1: %d → %d (ReLU)\n", INPUT_SIZE, HIDDEN1);
    printf("  Layer2: %d → %d (ReLU)\n", HIDDEN1, HIDDEN2);
    printf("  Layer3: %d → %d (Softmax)\n", HIDDEN2, NUM_CLASSES);

    printf("\n[2/4] Loading test images from %s...\n", images_path.c_str());

    std::vector<float> test_images(NUM_TEST * INPUT_SIZE);
    if (!read_binary(images_path.c_str(), test_images.data(), NUM_TEST * INPUT_SIZE * sizeof(float))) {
        return 1;
    }
    printf("  Loaded %d test images (%d features each)\n", NUM_TEST, INPUT_SIZE);

    printf("\n[3/4] Loading test labels from %s...\n", labels_path.c_str());

    std::vector<int32_t> test_labels(NUM_TEST);
    if (!read_binary(labels_path.c_str(), test_labels.data(), NUM_TEST * sizeof(int32_t))) {
        return 1;
    }
    printf("  Loaded %d test labels\n", NUM_TEST);

    printf("\n[4/4] Running inference on %d images...\n", NUM_TEST);

    std::vector<int> predictions(NUM_TEST);
    std::vector<float> confidences(NUM_TEST);

    auto t_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < NUM_TEST; i++) {

        // Non-owning view into test_images.
        Tensor input(test_images.data() + i * INPUT_SIZE, 1, INPUT_SIZE);

        Tensor h1 = layer1.forward_relu(input);
        Tensor h2 = layer2.forward_relu(h1);
        Tensor logits = layer3.forward_linear(h2);

        float probs[NUM_CLASSES];
        softmax(logits.data, probs, NUM_CLASSES);

        int pred = 0;
        float max_prob = probs[0];
        for (int j = 1; j < NUM_CLASSES; j++) {
            if (probs[j] > max_prob) {
                max_prob = probs[j];
                pred = j;
            }
        }

        predictions[i] = pred;
        confidences[i] = max_prob;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    int correct = 0;
    for (int i = 0; i < NUM_TEST; i++) {
        if (predictions[i] == test_labels[i]) {
            correct++;
        }
    }
    float accuracy = 100.0f * correct / NUM_TEST;

    printf("  Inference complete in %.2f ms\n", elapsed_ms);
    printf("  Throughput: %.1f images/sec\n", NUM_TEST / (elapsed_ms / 1000.0));
    printf("  Accuracy: %d/%d = %.1f%%\n", correct, NUM_TEST, accuracy);

    printf("\n─── Sample Predictions (first 10) ───────────────────────────\n");
    printf("  %-4s  %-12s  %-12s  %-8s\n", "Img#", "Predicted", "Actual", "Correct");
    printf("  %-4s  %-12s  %-12s  %-8s\n", "────", "──────────", "──────────", "───────");
    for (int i = 0; i < 10 && i < NUM_TEST; i++) {
        const char* pred_name = CLASS_NAMES[predictions[i]];
        const char* true_name = CLASS_NAMES[test_labels[i]];
        bool is_correct = (predictions[i] == test_labels[i]);
        printf("  %-4d  %-12s  %-12s  %s\n", i + 1, pred_name, true_name, is_correct ? "✓" : "✗");
    }

    printf("\n─── Writing results to %s ───\n", results_path.c_str());

    FILE* fout = fopen(results_path.c_str(), "w");
    if (!fout) {
        fprintf(stderr, "ERROR: Cannot create results file: %s\n", results_path.c_str());
        return 1;
    }

    fprintf(fout, "═══════════════════════════════════════════════════════════════\n");
    fprintf(fout, "Fashion MNIST C++ Inference Results\n");
    fprintf(fout, "═══════════════════════════════════════════════════════════════\n\n");

    fprintf(fout, "Model Architecture: %d → %d (ReLU) → %d (ReLU) → %d (Softmax)\n",
            INPUT_SIZE, HIDDEN1, HIDDEN2, NUM_CLASSES);
    fprintf(fout, "Test Images: %d\n", NUM_TEST);
    fprintf(fout, "Inference Time: %.2f ms\n", elapsed_ms);
    fprintf(fout, "Throughput: %.1f images/sec\n\n", NUM_TEST / (elapsed_ms / 1000.0));

    fprintf(fout, "Final Accuracy: %d/%d = %.1f%%\n\n", correct, NUM_TEST, accuracy);

    fprintf(fout, "───────────────────────────────────────────────────────────────\n");
    fprintf(fout, "Sample Predictions (first 10 images)\n");
    fprintf(fout, "───────────────────────────────────────────────────────────────\n\n");

    fprintf(fout, "  %-4s  %-12s  %-12s  %-8s  %s\n", "Img#", "Predicted", "Actual", "Correct", "Confidence");
    fprintf(fout, "  %-4s  %-12s  %-12s  %-8s  %s\n", "────", "──────────", "──────────", "───────", "──────────");

    for (int i = 0; i < 10 && i < NUM_TEST; i++) {
        const char* pred_name = CLASS_NAMES[predictions[i]];
        const char* true_name = CLASS_NAMES[test_labels[i]];
        bool is_correct = (predictions[i] == test_labels[i]);
        fprintf(fout, "  %-4d  %-12s  %-12s  %s       %.1f%%\n",
                i + 1, pred_name, true_name, is_correct ? "✓" : "✗", confidences[i] * 100.0f);
    }

    fprintf(fout, "\n───────────────────────────────────────────────────────────────\n");
    fprintf(fout, "Full Accuracy Breakdown by Class\n");
    fprintf(fout, "───────────────────────────────────────────────────────────────\n\n");

    int class_correct[NUM_CLASSES] = {};
    int class_total[NUM_CLASSES] = {};
    for (int i = 0; i < NUM_TEST; i++) {
        class_total[test_labels[i]]++;
        if (predictions[i] == test_labels[i]) {
            class_correct[test_labels[i]]++;
        }
    }

    fprintf(fout, "  %-12s  %-8s  %-8s  %-8s\n", "Class", "Total", "Correct", "Accuracy");
    fprintf(fout, "  %-12s  %-8s  %-8s  %-8s\n", "──────────", "──────", "───────", "────────");
    for (int c = 0; c < NUM_CLASSES; c++) {
        float class_acc = class_total[c] > 0 ? 100.0f * class_correct[c] / class_total[c] : 0.0f;
        fprintf(fout, "  %-12s  %-8d  %-8d  %.1f%%\n",
                CLASS_NAMES[c], class_total[c], class_correct[c], class_acc);
    }

    fprintf(fout, "\n═══════════════════════════════════════════════════════════════\n");
    fprintf(fout, "OVERALL ACCURACY: %.1f%%\n", accuracy);
    fprintf(fout, "═══════════════════════════════════════════════════════════════\n");

    fclose(fout);

    printf("  ✓ Results saved to: %s\n", results_path.c_str());

    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("SUCCESS! Inference complete.\n");
    printf("═══════════════════════════════════════════════════════════════\n");

    return 0;
}
