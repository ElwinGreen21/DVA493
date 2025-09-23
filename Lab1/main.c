#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define NUM_FEATURES 16
#define NUM_OUTPUTS 2
#define NUM_HIDDEN 16   // du kan experimentera med storlek


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

// ReLU aktiveringsfunktion. Intervall som i fallet med sigmoid är [0, ∞)
double relu(double x) {
    return (x > 0) ? x : 0.0;
}

// Derivata av ReLU
double relu_derivative(double x) {
    return (x > 0) ? 1.0 : 0.0;
}

// Framåtpropagering
void forward_propagation(double *x, double Weight_input_hidden[NUM_HIDDEN][NUM_FEATURES], double *bias_hidden, double Weight_hidden_output[NUM_OUTPUTS][NUM_HIDDEN], double *bias_outputs, double *hidden, double *outputs) {
    // Steg 1: Input → Hidden
    for (int i = 0; i < NUM_HIDDEN; i++) {
        double sum = bias_hidden[i];
        for (int j = 0; j < NUM_FEATURES; j++) {
            sum += x[j] * Weight_input_hidden[i][j];
        }
        hidden[i] = relu(sum);
    }

    // Steg 2: Hidden → Output
    for (int i = 0; i < NUM_OUTPUTS; i++) {
        double sum = bias_outputs[i];
        for (int j = 0; j < NUM_HIDDEN; j++) {
            sum += hidden[j] *  Weight_hidden_output[i][j];
        }
        outputs[i] = relu(sum);
    }
}

// Mean Squared Error
double mean_squared_error(double *y_true, double *y_pred, int size) {
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        double diff = y_true[i] - y_pred[i];
        sum += diff * diff;
    }
    return sum / size;
}


void back_propagation(double *x, double *y_true, double Weight_input_hidden[NUM_HIDDEN][NUM_FEATURES], double *bias_hidden, double Weight_hidden_output[NUM_OUTPUTS][NUM_HIDDEN], double *bias_outputs, double *hidden, double *outputs, double *z_hidden, double *z_output, double learning_rate) {
    double delta_output[NUM_OUTPUTS];
    double delta_hidden[NUM_HIDDEN];

    // --- Steg 1: Output layer error ---
    for (int i = 0; i < NUM_OUTPUTS; i++) {
        double error = outputs[i] - y_true[i]; // dL/dy
        delta_output[i] = error * relu_derivative(z_output[i]); // dL/dz
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

    // Initialize neural network parameters
    double hidden[NUM_HIDDEN];
    double outputs[NUM_OUTPUTS] = {0.0, 0.0};

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

    //Forward propagation for all training data
    for(int i = 0; i < datasets.train.size; i++) {

        forward_propagation(datasets.train.X[i], Weight_input_hidden, bias_hidden, Weight_hidden_output, bias_outputs, hidden, outputs);
    }
    //En epoch = alla träningsrader en gång (forward + backward + update).
    
    printf("Output 1: %f\n", outputs[0]);
    printf("Output 2: %f\n", outputs[1]);



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