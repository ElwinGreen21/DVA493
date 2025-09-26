#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define NUM_FEATURES 16
#define NUM_OUTPUTS 2
#define NUM_HIDDEN 16   // du kan experimentera med storlek
//#define learning_rate 0.001

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


void forward_propagation(double *x, double Weight_input_hidden[NUM_HIDDEN][NUM_FEATURES], double *bias_hidden, double Weight_hidden_output[NUM_OUTPUTS][NUM_HIDDEN], double *bias_outputs, double *hidden, double *outputs, double *z_hidden, double *z_output) {
    // Steg 1: Input → Hidden
    for (int i = 0; i < NUM_HIDDEN; i++) {
        double sum = bias_hidden[i];
        for (int j = 0; j < NUM_FEATURES; j++) {
            sum += x[j] * Weight_input_hidden[i][j];
        }
        z_hidden[i] = sum;
        hidden[i] = relu(sum);
    }

    // Steg 2: Hidden → Output
    for (int i = 0; i < NUM_OUTPUTS; i++) {
        double sum = bias_outputs[i];
        for (int j = 0; j < NUM_HIDDEN; j++) {
            sum += hidden[j] *  Weight_hidden_output[i][j];
        }
        z_output[i] = sum;
        outputs[i] = sigmoid(sum);
    }
}

void mse_per_output(double *y_true, double *y_pred, int size, double *loss_per_output) {
    for (int i = 0; i < size; i++) {
        double diff = y_true[i] - y_pred[i];
        loss_per_output[i] = diff * diff;
    }
}



void back_propagation(double *x, double *y_true, double Weight_input_hidden[NUM_HIDDEN][NUM_FEATURES], double *bias_hidden, double Weight_hidden_output[NUM_OUTPUTS][NUM_HIDDEN], double *bias_outputs, double *hidden, double *outputs, double *z_hidden, double *z_output, double learning_rate) {
    double delta_output[NUM_OUTPUTS];
    double delta_hidden[NUM_HIDDEN];

    // --- Steg 1: Output layer error ---
    for (int i = 0; i < NUM_OUTPUTS; i++) {
        double error = outputs[i] - y_true[i]; // dL/dy
        delta_output[i] = error * sigmoid_derivative(z_output[i]);
    }

    // --- Steg 2: Hidden layer error ---
    for (int j = 0; j < NUM_HIDDEN; j++) {
        double sum = 0.0;
        for (int i = 0; i < NUM_OUTPUTS; i++) {
            sum += delta_output[i] * Weight_hidden_output[i][j];
        }
        delta_hidden[j] = sum * relu_derivative(z_hidden[j]);
    }

    // --- Steg 3: Update weights Hidden → Output ---
    for (int i = 0; i < NUM_OUTPUTS; i++) {
        for (int j = 0; j < NUM_HIDDEN; j++) {
            Weight_hidden_output[i][j] -= learning_rate * delta_output[i] * hidden[j];
        }
        bias_outputs[i] -= learning_rate * delta_output[i];
    }

    // --- Steg 4: Update weights Input → Hidden ---
    for (int j = 0; j < NUM_HIDDEN; j++) {
        for (int k = 0; k < NUM_FEATURES; k++) {
            Weight_input_hidden[j][k] -= learning_rate * delta_hidden[j] * x[k];
        }
        bias_hidden[j] -= learning_rate * delta_hidden[j];
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
    double hidden[NUM_HIDDEN];
    double outputs[NUM_OUTPUTS];

    double Weight_input_hidden[NUM_HIDDEN][NUM_FEATURES];  // från input (16) till hidden (8)
    double bias_hidden[NUM_HIDDEN];   // bias för hidden layer

    double Weight_hidden_output[NUM_OUTPUTS][NUM_HIDDEN];   // från hidden (8) till output (2)
    double bias_outputs[NUM_OUTPUTS];  // bias för output

    // Initiera vikter slumpmässigt (exempel: -0.5 till 0.5)
    for (int i = 0; i < NUM_HIDDEN; i++) {
        bias_hidden[i] = ((double) rand() / RAND_MAX) - 0.5;

        for (int j = 0; j < NUM_FEATURES; j++) {
            Weight_input_hidden[i][j] = ((double) rand() / RAND_MAX) - 0.5;
        }
    }

    for (int i = 0; i < NUM_OUTPUTS; i++) {
        bias_outputs[i] = ((double) rand() / RAND_MAX) - 0.5;
        
        for (int j = 0; j < NUM_HIDDEN; j++) {
            Weight_hidden_output[i][j] = ((double) rand() / RAND_MAX) - 0.5;
        }
    }


    int epochs = 1000;  // hur många varv datan tränar
    double base_lr = 0.001;   // start learning rate
    double decay = 0.95;  // 5% minskning per epoch

    // Buffertar för forward/backward
    double z_hidden[NUM_HIDDEN];
    double z_output[NUM_OUTPUTS];  
    double loss_per_output[NUM_OUTPUTS];
    

    for (int epoch = 0; epoch < epochs; epoch++) { //start epochs
        double current_lr = base_lr * pow(decay, epoch);
        double total_loss[] = {0.0, 0.0};

    // --- loopa över alla rader i träningsdatan ---
    for (int i = 0; i < datasets.train.size; i++) {
        // ---- Forward ----
        forward_propagation(datasets.train.X[i], Weight_input_hidden, bias_hidden, Weight_hidden_output, bias_outputs, hidden, outputs, z_hidden, z_output);

        // ---- Loss ---- fixat
        mse_per_output(datasets.train.y[i], outputs, NUM_OUTPUTS, loss_per_output);
        total_loss[0] += loss_per_output[0];
        total_loss[1] += loss_per_output[1];

        // ---- Backward ----
        back_propagation(datasets.train.X[i], datasets.train.y[i], Weight_input_hidden, bias_hidden, Weight_hidden_output, bias_outputs,hidden, outputs,z_hidden, z_output, current_lr);
    }

    // skriv ut snitt-loss för den här epoken
    printf("Epoch %d, Compressor Loss: %f, Turbine Loss: %f\n", epoch + 1,  total_loss[0]/ datasets.train.size, total_loss[1]/ datasets.train.size);

        // --- Valideringsloss ---
    double val_loss_outputs[NUM_OUTPUTS] = {0};

    for (int i = 0; i < datasets.val.size; i++) {
        forward_propagation(datasets.val.X[i], Weight_input_hidden, bias_hidden, Weight_hidden_output, bias_outputs, hidden, outputs, z_hidden, z_output);

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
        forward_propagation(datasets.test.X[i], Weight_input_hidden, bias_hidden, Weight_hidden_output, bias_outputs, hidden, outputs, z_hidden, z_output);

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