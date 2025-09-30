#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define NUM_FEATURES 16
#define NUM_OUTPUTS 2
#define NUM_HIDDEN1 256   // du kan experimentera med storlek
#define NUM_HIDDEN2 128   // andra hidden-lagret, välj storlek
#define NUM_HIDDEN3 64
#define NUM_HIDDEN4 32
typedef struct {
    double **X;   // features
    double **y;   // labels
    int size;     // antal rader
} Dataset;

typedef struct {
    Dataset train;
    Dataset val;
    Dataset test;
} Split;


//Vi behöver minnst 3 funktioner, en för att läsa in data, en för att träna modellen och en för att göra prediktioner.


void normalize_data(Split *datasets) {

    double output_min[NUM_OUTPUTS];
    double output_max[NUM_OUTPUTS];    

    // Initiera min/max
    for (int j = 0; j < NUM_OUTPUTS; j++) {
        output_min[j] = datasets->train.y[0][j];
        output_max[j] = datasets->train.y[0][j];
    }

    // Hitta min/max
    for (int i = 0; i < datasets->train.size; i++) {
        for (int j = 0; j < NUM_OUTPUTS; j++) {
            if (datasets->train.y[i][j] < output_min[j]) output_min[j] = datasets->train.y[i][j];
            if (datasets->train.y[i][j] > output_max[j]) output_max[j] = datasets->train.y[i][j];
        }
    }

    // Normalisera train
    for (int i = 0; i < datasets->train.size; i++) {
        for (int j = 0; j < NUM_OUTPUTS; j++) {
            datasets->train.y[i][j] =
                (datasets->train.y[i][j] - output_min[j]) / (output_max[j] - output_min[j]);
        }
    }

    // Normalisera val
    for (int i = 0; i < datasets->val.size; i++) {
        for (int j = 0; j < NUM_OUTPUTS; j++) {
            datasets->val.y[i][j] =
                (datasets->val.y[i][j] - output_min[j]) / (output_max[j] - output_min[j]);
        }
    }

    // Normalisera test
    for (int i = 0; i < datasets->test.size; i++) {
        for (int j = 0; j < NUM_OUTPUTS; j++) {
            datasets->test.y[i][j] =
                (datasets->test.y[i][j] - output_min[j]) / (output_max[j] - output_min[j]);
        }
    }

    double feature_min[NUM_FEATURES];
    double feature_max[NUM_FEATURES];

    // 1. Initiera min/max
    for (int j = 0; j < NUM_FEATURES; j++) {
        feature_min[j] = datasets->train.X[0][j];
        feature_max[j] = datasets->train.X[0][j];
    }

    // 2. Hitta min och max över hela datasetet
    for (int i = 0; i < datasets->train.size; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            if (datasets->train.X[i][j] < feature_min[j]) feature_min[j] = datasets->train.X[i][j];
            if (datasets->train.X[i][j] > feature_max[j]) feature_max[j] = datasets->train.X[i][j];
        }
    }


    // Normalisera train
    for (int i = 0; i < datasets->train.size; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            datasets->train.X[i][j] =
                (datasets->train.X[i][j] - feature_min[j]) / (feature_max[j] - feature_min[j]);
        }
    }

    // Normalisera val
    for (int i = 0; i < datasets->val.size; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            datasets->val.X[i][j] =
                (datasets->val.X[i][j] - feature_min[j]) / (feature_max[j] - feature_min[j]);
        }
    }

    // Normalisera test
    for (int i = 0; i < datasets->test.size; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            datasets->test.X[i][j] =
                (datasets->test.X[i][j] - feature_min[j]) / (feature_max[j] - feature_min[j]);
        }
    }
}

// ReLU aktiveringsfunktion
double relu(double x) {
    return (x > 0) ? x : 0.0;
}

// Derivata av ReLU
double relu_derivative(double x) {
    return (x > 0) ? 1.0 : 0.0;
}

// Sigmoid funktion
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Derivatan
double sigmoid_derivative(double x) {
    double s = sigmoid(x);
    return s * (1 - s);
}


void forward_propagation(double *x, double W_input_h1[NUM_HIDDEN1][NUM_FEATURES], double b_h1[NUM_HIDDEN1], double W_h1_h2[NUM_HIDDEN2][NUM_HIDDEN1], double b_h2[NUM_HIDDEN2], double W_h2_h3[NUM_HIDDEN3][NUM_HIDDEN2], double b_h3[NUM_HIDDEN3], double W_h3_h4[NUM_HIDDEN4][NUM_HIDDEN3], double b_h4[NUM_HIDDEN4], double W_h4_out[NUM_OUTPUTS][NUM_HIDDEN4], double b_out[NUM_OUTPUTS], double h1[NUM_HIDDEN1], double h2[NUM_HIDDEN2], double h3[NUM_HIDDEN3], double h4[NUM_HIDDEN4], double outputs[NUM_OUTPUTS], double z_h1[NUM_HIDDEN1], double z_h2[NUM_HIDDEN2], double z_h3[NUM_HIDDEN3], double z_h4[NUM_HIDDEN4], double z_out[NUM_OUTPUTS]) {
    // Input -> H1
    for (int i = 0; i < NUM_HIDDEN1; i++) {
        double sum = b_h1[i];
        for (int j = 0; j < NUM_FEATURES; j++)
            sum += x[j] * W_input_h1[i][j];
        z_h1[i] = sum;
        h1[i] = relu(sum);
    }

    // H1 -> H2
    for (int i = 0; i < NUM_HIDDEN2; i++) {
        double sum = b_h2[i];
        for (int j = 0; j < NUM_HIDDEN1; j++)
            sum += h1[j] * W_h1_h2[i][j];
        z_h2[i] = sum;
        h2[i] = relu(sum);
    }

    // H2 -> H3
    for (int i = 0; i < NUM_HIDDEN3; i++) {
        double sum = b_h3[i];
        for (int j = 0; j < NUM_HIDDEN2; j++)
            sum += h2[j] * W_h2_h3[i][j];
        z_h3[i] = sum;
        h3[i] = relu(sum);
    }

    // H3 -> H4
    for (int i = 0; i < NUM_HIDDEN4; i++) {
        double sum = b_h4[i];
        for (int j = 0; j < NUM_HIDDEN3; j++)
            sum += h3[j] * W_h3_h4[i][j];
        z_h4[i] = sum;
        h4[i] = relu(sum);
    }

    // H4 -> Output
    for (int i = 0; i < NUM_OUTPUTS; i++) {
        double sum = b_out[i];
        for (int j = 0; j < NUM_HIDDEN4; j++)
            sum += h4[j] * W_h4_out[i][j];
        z_out[i] = sum;
        outputs[i] = sigmoid(sum); // regression kan ev. använda linjär aktivering
    }
}

void mse_per_output(double *y_true, double *y_pred, int size, double *loss_per_output) {
    for (int i = 0; i < size; i++) {
        double diff = y_true[i] - y_pred[i];
        loss_per_output[i] = diff * diff;
    }
}



void back_propagation(double *x, double *y_true, double W_input_h1[NUM_HIDDEN1][NUM_FEATURES], double b_h1[NUM_HIDDEN1], double W_h1_h2[NUM_HIDDEN2][NUM_HIDDEN1], double b_h2[NUM_HIDDEN2], double W_h2_h3[NUM_HIDDEN3][NUM_HIDDEN2], double b_h3[NUM_HIDDEN3], double W_h3_h4[NUM_HIDDEN4][NUM_HIDDEN3], double b_h4[NUM_HIDDEN4], double W_h4_out[NUM_OUTPUTS][NUM_HIDDEN4], double b_out[NUM_OUTPUTS], double h1[NUM_HIDDEN1], double h2[NUM_HIDDEN2], double h3[NUM_HIDDEN3], double h4[NUM_HIDDEN4], double outputs[NUM_OUTPUTS], double z_h1[NUM_HIDDEN1], double z_h2[NUM_HIDDEN2], double z_h3[NUM_HIDDEN3], double z_h4[NUM_HIDDEN4], double z_out[NUM_OUTPUTS], double learning_rate) {  
    double delta_out[NUM_OUTPUTS];
    double delta_h4[NUM_HIDDEN4];
    double delta_h3[NUM_HIDDEN3];
    double delta_h2[NUM_HIDDEN2];
    double delta_h1[NUM_HIDDEN1];

    // Output
    for (int i = 0; i < NUM_OUTPUTS; i++) {
        double error = outputs[i] - y_true[i];
        delta_out[i] = error * sigmoid_derivative(z_out[i]);
    }

    // H4
    for (int j = 0; j < NUM_HIDDEN4; j++) {
        double sum = 0;
        for (int i = 0; i < NUM_OUTPUTS; i++)
            sum += delta_out[i] * W_h4_out[i][j];
        delta_h4[j] = sum * relu_derivative(z_h4[j]);
    }

    // H3
    for (int j = 0; j < NUM_HIDDEN3; j++) {
        double sum = 0;
        for (int k = 0; k < NUM_HIDDEN4; k++)
            sum += delta_h4[k] * W_h3_h4[k][j];
        delta_h3[j] = sum * relu_derivative(z_h3[j]);
    }

    // H2
    for (int j = 0; j < NUM_HIDDEN2; j++) {
        double sum = 0;
        for (int k = 0; k < NUM_HIDDEN3; k++)
            sum += delta_h3[k] * W_h2_h3[k][j];
        delta_h2[j] = sum * relu_derivative(z_h2[j]);
    }

    // H1
    for (int j = 0; j < NUM_HIDDEN1; j++) {
        double sum = 0;
        for (int k = 0; k < NUM_HIDDEN2; k++)
            sum += delta_h2[k] * W_h1_h2[k][j];
        delta_h1[j] = sum * relu_derivative(z_h1[j]);
    }

    // H4 -> Output
    for (int i = 0; i < NUM_OUTPUTS; i++) {
        for (int j = 0; j < NUM_HIDDEN4; j++)
            W_h4_out[i][j] -= learning_rate * delta_out[i] * h4[j];
        b_out[i] -= learning_rate * delta_out[i];
    }

    // H3 -> H4
    for (int j = 0; j < NUM_HIDDEN4; j++) {
        for (int k = 0; k < NUM_HIDDEN3; k++)
            W_h3_h4[j][k] -= learning_rate * delta_h4[j] * h3[k];
        b_h4[j] -= learning_rate * delta_h4[j];
    }

    // H2 -> H3
    for (int j = 0; j < NUM_HIDDEN3; j++) {
        for (int k = 0; k < NUM_HIDDEN2; k++)
            W_h2_h3[j][k] -= learning_rate * delta_h3[j] * h2[k];
        b_h3[j] -= learning_rate * delta_h3[j];
    }

    // --- H1 -> H2 ---
    for (int i = 0; i < NUM_HIDDEN2; i++) {
        for (int j = 0; j < NUM_HIDDEN1; j++) {
            W_h1_h2[i][j] -= learning_rate* delta_h2[i] * h1[j];
        }
        b_h2[i] -= learning_rate * delta_h2[i];
    }

    // --- Input -> H1 ---
    for (int i = 0; i < NUM_HIDDEN1; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            W_input_h1[i][j] -= learning_rate * delta_h1[i] * x[j];
        }
        b_h1[i] -= learning_rate * delta_h1[i];
    }
}


// randomisera data för att undvika bias och dela sedan upp i träning,test,validering
// Fisher-Yates shuffle algorithm
Split shuffle_data(int num_rows, double **X, double **y) {
    for (int i = num_rows -1; i > 0; i--){
        int j = rand() % (i + 1);

        double* tempX = X[i];
        X[i] = X[j];
        X[j] = tempX;

        double* tempY = y[i];
        y[i] = y[j];
        y[j] = tempY;
    } 

    int train_size = num_rows * 0.5;
    int val_size = num_rows * 0.25;
    int test_size = num_rows - train_size - val_size;

    Split split = {
        { X, y, train_size },
        { X + train_size, y + train_size, val_size },
        { X + train_size + val_size, y + train_size + val_size, test_size }
    };

    return split;
}

int main(void) {
    srand(time(NULL));

    // Open file
    FILE *file = fopen("maintenance.txt", "r");
    if (!file) {
        printf("Could not open maintenance.txt\n");
        return 1;
    }

    // Count lines
    int num_rows = 0;
    char ch;
    while (!feof(file)) {
        ch = fgetc(file);
        if (ch == '\n') num_rows++;
    }
    rewind(file);

    // Allocate memory
    double **X = malloc(num_rows * sizeof(double*));
    double **y = malloc(num_rows * sizeof(double*));
    for (int i = 0; i < num_rows; i++) {
        X[i] = malloc(NUM_FEATURES * sizeof(double));
        y[i] = malloc(NUM_OUTPUTS * sizeof(double));
    }

    // Read data
    int row = 0;
    while (row < num_rows) {
        int read = fscanf(
            file,
            "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
            &X[row][0], &X[row][1], &X[row][2], &X[row][3], &X[row][4], &X[row][5], &X[row][6], &X[row][7],
            &X[row][8], &X[row][9], &X[row][10], &X[row][11], &X[row][12], &X[row][13], &X[row][14], &X[row][15],
            &y[row][0], &y[row][1]
        );
        if (read != NUM_FEATURES + NUM_OUTPUTS) break;
        row++;
    }

    printf("Read %d rows\n", row);

    Split datasets = shuffle_data(num_rows, X, y);
    normalize_data(&datasets);

    // Initialize neural network parameters
    double hidden[NUM_HIDDEN1];
    double outputs[NUM_OUTPUTS];

    // Input -> H1
    double W_input_h1[NUM_HIDDEN1][NUM_FEATURES];
    double b_h1[NUM_HIDDEN1];

    // H1 -> H2
    double W_h1_h2[NUM_HIDDEN2][NUM_HIDDEN1];
    double b_h2[NUM_HIDDEN2];

    // H2 -> H3
    double W_h2_h3[NUM_HIDDEN3][NUM_HIDDEN2];
    double b_h3[NUM_HIDDEN3];

    // H3 -> H4
    double W_h3_h4[NUM_HIDDEN4][NUM_HIDDEN3];
    double b_h4[NUM_HIDDEN4];

    // H4 -> Output
    double W_h4_out[NUM_OUTPUTS][NUM_HIDDEN4];
    double b_out[NUM_OUTPUTS];
    
    // ---buffert---
    double z_h1[NUM_HIDDEN1], h1[NUM_HIDDEN1];
    double z_h2[NUM_HIDDEN2], h2[NUM_HIDDEN2];
    double z_h3[NUM_HIDDEN3], h3[NUM_HIDDEN3];
    double z_h4[NUM_HIDDEN4], h4[NUM_HIDDEN4];
    double z_out[NUM_OUTPUTS], outputs[NUM_OUTPUTS];


    // Initiera vikter slumpmässigt (exempel: -0.5 till 0.5)
    for (int i = 0; i < NUM_HIDDEN1; i++) {
        b_h1[i] = ((double) rand() / RAND_MAX) - 0.5;
        for (int j = 0; j < NUM_FEATURES; j++)
            W_input_h1[i][j] = ((double) rand() / RAND_MAX) - 0.5;
    }

    for (int i = 0; i < NUM_HIDDEN2; i++) {
        b_h2[i] = ((double) rand() / RAND_MAX) - 0.5;
        for (int j = 0; j < NUM_HIDDEN1; j++)
            W_h1_h2[i][j] = ((double) rand() / RAND_MAX) - 0.5;
    }

    for (int i = 0; i < NUM_HIDDEN3; i++) {
        b_h3[i] = ((double) rand() / RAND_MAX) - 0.5;
        for (int j = 0; j < NUM_HIDDEN2; j++)
            W_h2_h3[i][j] = ((double) rand() / RAND_MAX) - 0.5;
    }

    for (int i = 0; i < NUM_HIDDEN4; i++) {
        b_h4[i] = ((double) rand() / RAND_MAX) - 0.5;
        for (int j = 0; j < NUM_HIDDEN3; j++)
            W_h3_h4[i][j] = ((double) rand() / RAND_MAX) - 0.5;
    }

    for (int i = 0; i < NUM_OUTPUTS; i++) {
        b_out[i] = ((double) rand() / RAND_MAX) - 0.5;
        for (int j = 0; j < NUM_HIDDEN4; j++)
            W_h4_out[i][j] = ((double) rand() / RAND_MAX) - 0.5;
    }


    int epochs = 1000;  // hur många varv datan tränar
    double base_lr = 0.001;   // start learning rate
    double decay = 0.99;  // 5% minskning per epoch

    // Buffertar för forward/backward
   
    double loss_per_output[NUM_OUTPUTS];
    

    for (int epoch = 0; epoch < epochs; epoch++) { //start epochs
        double current_lr = base_lr * pow(decay, epoch);
        double total_loss[] = {0.0, 0.0};

    // --- loopa över alla rader i träningsdatan ---
    for (int i = 0; i < datasets.train.size; i++) {
        // ---- Forward ----
        forward_propagation(datasets.train.X[i], W_input_h1, b_h1, W_h1_h2, b_h2, W_h2_h3, b_h3, W_h3_h4, b_h4, W_h4_out, b_out, h1, h2, h3, h4, outputs, z_h1, z_h2, z_h3, z_h4, z_out);

        // ---- Loss ---- fixat
        mse_per_output(datasets.train.y[i], outputs, NUM_OUTPUTS, loss_per_output);
        total_loss[0] += loss_per_output[0];
        total_loss[1] += loss_per_output[1];

        // ---- Backward ----
        back_propagation(datasets.train.X[i], datasets.train.y[i], W_input_h1, b_h1, W_h1_h2, b_h2, W_h2_h3, b_h3, W_h3_h4, b_h4, W_h4_out, b_out, h1, h2, h3, h4, outputs, z_h1, z_h2, z_h3, z_h4, z_out, current_lr);
    }

    // skriv ut snitt-loss för den här epoken
    printf("Epoch %d, Compressor Loss: %f, Turbine Loss: %f\n", epoch + 1,  total_loss[0]/ datasets.train.size, total_loss[1]/ datasets.train.size);

        // --- Valideringsloss ---
    double val_loss_outputs[NUM_OUTPUTS] = {0};

    for (int i = 0; i < datasets.val.size; i++) {
        forward_propagation(datasets.val.X[i], W_input_h1, b_h1, W_h1_h2, b_h2, W_h2_h3, b_h3, W_h3_h4, b_h4, W_h4_out, b_out, h1, h2, h3, h4, outputs, z_h1, z_h2, z_h3, z_h4, z_out);

        mse_per_output(datasets.val.y[i], outputs, NUM_OUTPUTS, loss_per_output);

        for (int k = 0; k < NUM_OUTPUTS; k++) {
            val_loss_outputs[k] += loss_per_output[k];
        }
    }

    printf("Validation Loss -> Compressor: %f, Turbine: %f\n", val_loss_outputs[0] / datasets.val.size, val_loss_outputs[1] / datasets.val.size);

    } // end epochs

        // --- Testloss (slutlig utvärdering) ---
    double test_loss_outputs[NUM_OUTPUTS] = {0};

    for (int i = 0; i < datasets.test.size; i++) {
        forward_propagation(datasets.test.X[i], W_input_h1, b_h1, W_h1_h2, b_h2, W_h2_h3, b_h3, W_h3_h4, b_h4, W_h4_out, b_out, h1, h2, h3, h4, outputs, z_h1, z_h2, z_h3, z_h4, z_out);

        mse_per_output(datasets.test.y[i], outputs, NUM_OUTPUTS, loss_per_output);

        for (int k = 0; k < NUM_OUTPUTS; k++) {
            test_loss_outputs[k] += loss_per_output[k];
        }
    }

    printf("Final Test Loss -> Compressor: %f, Turbine: %f\n",
        test_loss_outputs[0] / datasets.test.size,
        test_loss_outputs[1] / datasets.test.size);


    //En epoch = alla träningsrader en gång (forward + backward + update).
    
    //printf("Output 1: %f\n", outputs[0]);
    //printf("Output 2: %f\n", outputs[1]);

    // Free memory!!!!!!!!!!!!! ska nog köras sist i main
    for (int i = 0; i < num_rows; i++) {
        free(X[i]);
        free(y[i]);
    }
    free(X);
    free(y);

    fclose(file);
    return 0;
}