#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define NUM_FEATURES 16
#define NUM_OUTPUTS 2

int val_index;
int train_index;
int test_index;


//Vi behöver minnst 3 funktioner, en för att läsa in data, en för att träna modellen och en för att göra prediktioner.

// ReLU aktiveringsfunktion
double relu(double x) {
    return (x > 0) ? x : 0.0;
}

// Derivata av ReLU
double relu_derivative(double x) {
    return (x > 0) ? 1.0 : 0.0;
}

double forward_propagation(double* x, double* y, double weights[2][16]) {
    // Placeholder för framåtpropageringslogik
    double outputs[2] = {0.0, 0.0};
    for (int i = 0; i < 2; i++) {
        double sum = y[i];
        for (int j = 0; j < 16; j++) {
            sum += x[j] * weights[i][j];
        }
        outputs[i] = relu(sum);  // <-- här används ReLU
    }

    return outputs[0]; // Returnerar första output för enkelhetens skull
}

int back_propagation() {
    // Placeholder för bakåtpropageringslogik
    return 0;
}

// randomisera data för att undvika bias och dela sedan upp i träning,test,validering
// Fisher-Yates shuffle algorithm
int shuffle_data(int num_rows, double **X, double **y) {
    for (int i = num_rows -1; i > 0; i--){
        int j = rand() % (i + 1);

        double* tempX = X[i];
        X[i] = X[j];
        X[j] = tempX;

        double* tempY = y[i];
        y[i] = y[j];
        y[j] = tempY;
    } 

    train_index = num_rows * 0.5;
    val_index = num_rows * 0.25 + val_index;
    test_index = num_rows - train_index - val_index;
    return 0;
}

int main(void) {
    srand(time(NULL));

    double weights[2][16] = {
        {0.2, -0.1, 0.4, 0.3, -0.5, 0.1, 0.2, 0.7,
         0.3, -0.2, 0.1, 0.5, -0.4, 0.6, 0.2, -0.3},
        {-0.3, 0.6, 0.2, -0.1, 0.4, 0.5, -0.2, 0.1,
         0.7, 0.3, -0.6, 0.2, 0.1, -0.5, 0.4, 0.3}
    };

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

    // Free memory!!!!!!!!!!!!! ska nog köras sist i main
    for (int i = 0; i < num_rows; i++) {
        free(X[i]);
        free(y[i]);
    }
    free(X);
    free(y);

    fclose(file);

    double outputs[] = forward_propagation(X*, y*, weights**);
    
    printf("Output 1: %f\n", outputs[0]);
    printf("Output 2: %f\n", outputs[1]);
    return 0;
}