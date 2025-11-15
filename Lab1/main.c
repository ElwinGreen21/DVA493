#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define NUM_FEATURES 16
#define NUM_OUTPUTS 2
#define NUM_HIDDEN1 64
#define NUM_HIDDEN2 32
#define NUM_HIDDEN3 16
#define NUM_HIDDEN4 8

typedef struct {
    double **X;
    double **y;
    int size;
} Dataset;

typedef struct {
    Dataset train;
    Dataset val;
    Dataset test;
} Split;

// ------------------- Normalisering -------------------
void normalize_data(Split *datasets,
                    double output_min[NUM_OUTPUTS], double output_max[NUM_OUTPUTS],
                    double feature_min[NUM_FEATURES], double feature_max[NUM_FEATURES]) {

    // init output min/max
    for (int j = 0; j < NUM_OUTPUTS; j++) {
        output_min[j] = datasets->train.y[0][j];
        output_max[j] = datasets->train.y[0][j];
    }
    for (int i = 0; i < datasets->train.size; i++) {
        for (int j = 0; j < NUM_OUTPUTS; j++) {
            if (datasets->train.y[i][j] < output_min[j]) output_min[j] = datasets->train.y[i][j];
            if (datasets->train.y[i][j] > output_max[j]) output_max[j] = datasets->train.y[i][j];
        }
    }
    // normalisera outputs
    for (int i = 0; i < datasets->train.size; i++) {
        for (int j = 0; j < NUM_OUTPUTS; j++) {
            double denom = output_max[j] - output_min[j];
            datasets->train.y[i][j] = denom ? (datasets->train.y[i][j] - output_min[j]) / denom : 0.0;
        }
    }
    for (int i = 0; i < datasets->val.size; i++) {
        for (int j = 0; j < NUM_OUTPUTS; j++) {
            double denom = output_max[j] - output_min[j];
            datasets->val.y[i][j] = denom ? (datasets->val.y[i][j] - output_min[j]) / denom : 0.0;
        }
    }
    for (int i = 0; i < datasets->test.size; i++) {
        for (int j = 0; j < NUM_OUTPUTS; j++) {
            double denom = output_max[j] - output_min[j];
            datasets->test.y[i][j] = denom ? (datasets->test.y[i][j] - output_min[j]) / denom : 0.0;
        }
    }

    // init feature min/max
    for (int j = 0; j < NUM_FEATURES; j++) {
        feature_min[j] = datasets->train.X[0][j];
        feature_max[j] = datasets->train.X[0][j];
    }
    for (int i = 0; i < datasets->train.size; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            if (datasets->train.X[i][j] < feature_min[j]) feature_min[j] = datasets->train.X[i][j];
            if (datasets->train.X[i][j] > feature_max[j]) feature_max[j] = datasets->train.X[i][j];
        }
    }
    // normalisera features
    for (int i = 0; i < datasets->train.size; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            double denom = feature_max[j] - feature_min[j];
            datasets->train.X[i][j] = denom ? (datasets->train.X[i][j] - feature_min[j]) / denom : 0.0;
        }
    }
    for (int i = 0; i < datasets->val.size; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            double denom = feature_max[j] - feature_min[j];
            datasets->val.X[i][j] = denom ? (datasets->val.X[i][j] - feature_min[j]) / denom : 0.0;
        }
    }
    for (int i = 0; i < datasets->test.size; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            double denom = feature_max[j] - feature_min[j];
            datasets->test.X[i][j] = denom ? (datasets->test.X[i][j] - feature_min[j]) / denom : 0.0;
        }
    }
}

// ------------------- Aktiveringar -------------------
double relu(double x) { return (x > 0) ? x : 0.0; }
double relu_derivative(double x) { return (x > 0) ? 1.0 : 0.0; }

// ------------------- Forward -------------------
void forward_propagation(double *x,
    double W_input_h1[NUM_HIDDEN1][NUM_FEATURES], double b_h1[NUM_HIDDEN1],
    double W_h1_h2[NUM_HIDDEN2][NUM_HIDDEN1], double b_h2[NUM_HIDDEN2],
    double W_h2_h3[NUM_HIDDEN3][NUM_HIDDEN2], double b_h3[NUM_HIDDEN3],
    double W_h3_h4[NUM_HIDDEN4][NUM_HIDDEN3], double b_h4[NUM_HIDDEN4],
    double W_h4_out[NUM_OUTPUTS][NUM_HIDDEN4], double b_out[NUM_OUTPUTS],
    double h1[NUM_HIDDEN1], double h2[NUM_HIDDEN2],
    double h3[NUM_HIDDEN3], double h4[NUM_HIDDEN4],
    double outputs[NUM_OUTPUTS],
    double z_h1[NUM_HIDDEN1], double z_h2[NUM_HIDDEN2],
    double z_h3[NUM_HIDDEN3], double z_h4[NUM_HIDDEN4],
    double z_out[NUM_OUTPUTS]) {

    for (int i = 0; i < NUM_HIDDEN1; i++) {
        double sum = b_h1[i];
        for (int j = 0; j < NUM_FEATURES; j++) sum += x[j] * W_input_h1[i][j];
        z_h1[i] = sum; h1[i] = relu(sum);
    }
    for (int i = 0; i < NUM_HIDDEN2; i++) {
        double sum = b_h2[i];
        for (int j = 0; j < NUM_HIDDEN1; j++) sum += h1[j] * W_h1_h2[i][j];
        z_h2[i] = sum; h2[i] = relu(sum);
    }
    for (int i = 0; i < NUM_HIDDEN3; i++) {
        double sum = b_h3[i];
        for (int j = 0; j < NUM_HIDDEN2; j++) sum += h2[j] * W_h2_h3[i][j];
        z_h3[i] = sum; h3[i] = relu(sum);
    }
    for (int i = 0; i < NUM_HIDDEN4; i++) {
        double sum = b_h4[i];
        for (int j = 0; j < NUM_HIDDEN3; j++) sum += h3[j] * W_h3_h4[i][j];
        z_h4[i] = sum; h4[i] = relu(sum);
    }
    for (int i = 0; i < NUM_OUTPUTS; i++) {
        double sum = b_out[i];
        for (int j = 0; j < NUM_HIDDEN4; j++) sum += h4[j] * W_h4_out[i][j];
        z_out[i] = sum;
        outputs[i] = sum; // linjär output
    }
}

// ------------------- Loss -------------------
void mse_per_output(double *y_true, double *y_pred, int size, double *loss_per_output) {
    for (int i = 0; i < size; i++) {
        double diff = y_true[i] - y_pred[i];
        loss_per_output[i] = diff * diff;
    }
}

// ------------------- Backprop -------------------
void back_propagation(double *x, double *y_true,
    double W_input_h1[NUM_HIDDEN1][NUM_FEATURES], double b_h1[NUM_HIDDEN1],
    double W_h1_h2[NUM_HIDDEN2][NUM_HIDDEN1], double b_h2[NUM_HIDDEN2],
    double W_h2_h3[NUM_HIDDEN3][NUM_HIDDEN2], double b_h3[NUM_HIDDEN3],
    double W_h3_h4[NUM_HIDDEN4][NUM_HIDDEN3], double b_h4[NUM_HIDDEN4],
    double W_h4_out[NUM_OUTPUTS][NUM_HIDDEN4], double b_out[NUM_OUTPUTS],
    double h1[NUM_HIDDEN1], double h2[NUM_HIDDEN2],
    double h3[NUM_HIDDEN3], double h4[NUM_HIDDEN4],
    double outputs[NUM_OUTPUTS],
    double z_h1[NUM_HIDDEN1], double z_h2[NUM_HIDDEN2],
    double z_h3[NUM_HIDDEN3], double z_h4[NUM_HIDDEN4],
    double z_out[NUM_OUTPUTS], double learning_rate) {

    double delta_out[NUM_OUTPUTS];
    double delta_h4[NUM_HIDDEN4];
    double delta_h3[NUM_HIDDEN3];
    double delta_h2[NUM_HIDDEN2];
    double delta_h1[NUM_HIDDEN1];

    // Output (linjär)
    for (int i = 0; i < NUM_OUTPUTS; i++) {
        delta_out[i] = outputs[i] - y_true[i];
    }

    // H4
    for (int j = 0; j < NUM_HIDDEN4; j++) {
        double sum = 0;
        for (int i = 0; i < NUM_OUTPUTS; i++) sum += delta_out[i] * W_h4_out[i][j];
        delta_h4[j] = sum * relu_derivative(z_h4[j]);
    }
    // H3
    for (int j = 0; j < NUM_HIDDEN3; j++) {
        double sum = 0;
        for (int k = 0; k < NUM_HIDDEN4; k++) sum += delta_h4[k] * W_h3_h4[k][j];
        delta_h3[j] = sum * relu_derivative(z_h3[j]);
    }
    // H2
    for (int j = 0; j < NUM_HIDDEN2; j++) {
        double sum = 0;
        for (int k = 0; k < NUM_HIDDEN3; k++) sum += delta_h3[k] * W_h2_h3[k][j];
        delta_h2[j] = sum * relu_derivative(z_h2[j]);
    }
    // H1
    for (int j = 0; j < NUM_HIDDEN1; j++) {
        double sum = 0;
        for (int k = 0; k < NUM_HIDDEN2; k++) sum += delta_h2[k] * W_h1_h2[k][j];
        delta_h1[j] = sum * relu_derivative(z_h1[j]);
    }

    // Update weights
    for (int i = 0; i < NUM_OUTPUTS; i++) {
        for (int j = 0; j < NUM_HIDDEN4; j++)
            W_h4_out[i][j] -= learning_rate * delta_out[i] * h4[j];
        b_out[i] -= learning_rate * delta_out[i];
    }
    for (int j = 0; j < NUM_HIDDEN4; j++) {
        for (int k = 0; k < NUM_HIDDEN3; k++)
            W_h3_h4[j][k] -= learning_rate * delta_h4[j] * h3[k];
        b_h4[j] -= learning_rate * delta_h4[j];
    }
    for (int j = 0; j < NUM_HIDDEN3; j++) {
        for (int k = 0; k < NUM_HIDDEN2; k++)
            W_h2_h3[j][k] -= learning_rate * delta_h3[j] * h2[k];
        b_h3[j] -= learning_rate * delta_h3[j];
    }
    for (int i = 0; i < NUM_HIDDEN2; i++) {
        for (int j = 0; j < NUM_HIDDEN1; j++)
            W_h1_h2[i][j] -= learning_rate * delta_h2[i] * h1[j];
        b_h2[i] -= learning_rate * delta_h2[i];
    }
    for (int i = 0; i < NUM_HIDDEN1; i++) {
        for (int j = 0; j < NUM_FEATURES; j++)
            W_input_h1[i][j] -= learning_rate * delta_h1[i] * x[j];
        b_h1[i] -= learning_rate * delta_h1[i];
    }
}

// ------------------- Shuffle & split -------------------
Split shuffle_data(int num_rows, double **X, double **y) {
    for (int i = num_rows - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        double *tempX = X[i]; X[i] = X[j]; X[j] = tempX;
        double *tempY = y[i]; y[i] = y[j]; y[j] = tempY;
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

// ------------------- Hjälp -------------------
double rand_uniform() { return ((double)rand() / RAND_MAX) * 2.0 - 1.0; }
double denorm_output(double norm_val, double out_min, double out_max) {
    return norm_val * (out_max - out_min) + out_min;
}

// ------------------- MAIN -------------------
int main(void) {
    srand(time(NULL));
    FILE *file = fopen("maintenance.txt", "r");
    if (!file) { printf("Could not open maintenance.txt\n"); return 1; }

    FILE *val_log = fopen("val_loss_log.txt", "w");
    if (!val_log) {
        printf("Could not open val_loss_log.txt for writing\n");
        return 1;
    }

    // räkna rader
    int num_rows = 0; char ch;
    while (!feof(file)) { ch = fgetc(file); if (ch == '\n') num_rows++; }
    rewind(file);

    double **X = malloc(num_rows * sizeof(double*));
    double **y = malloc(num_rows * sizeof(double*));
    for (int i = 0; i < num_rows; i++) {
        X[i] = malloc(NUM_FEATURES * sizeof(double));
        y[i] = malloc(NUM_OUTPUTS * sizeof(double));
    }

    int row = 0;
    while (row < num_rows) {
        int read = fscanf(file,
            "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
            &X[row][0], &X[row][1], &X[row][2], &X[row][3], &X[row][4], &X[row][5], &X[row][6], &X[row][7],
            &X[row][8], &X[row][9], &X[row][10], &X[row][11], &X[row][12], &X[row][13], &X[row][14], &X[row][15],
            &y[row][0], &y[row][1]);
        if (read != NUM_FEATURES + NUM_OUTPUTS) break;
        row++;
    }
    printf("Read %d rows\n", row);

    Split datasets = shuffle_data(row, X, y);
    double output_min[NUM_OUTPUTS], output_max[NUM_OUTPUTS];
    double feature_min[NUM_FEATURES], feature_max[NUM_FEATURES];
    normalize_data(&datasets, output_min, output_max, feature_min, feature_max);

    // Vikter
    double W_input_h1[NUM_HIDDEN1][NUM_FEATURES], b_h1[NUM_HIDDEN1];
    double W_h1_h2[NUM_HIDDEN2][NUM_HIDDEN1], b_h2[NUM_HIDDEN2];
    double W_h2_h3[NUM_HIDDEN3][NUM_HIDDEN2], b_h3[NUM_HIDDEN3];
    double W_h3_h4[NUM_HIDDEN4][NUM_HIDDEN3], b_h4[NUM_HIDDEN4];
    double W_h4_out[NUM_OUTPUTS][NUM_HIDDEN4], b_out[NUM_OUTPUTS];

    // Buffertar
    double z_h1[NUM_HIDDEN1], h1[NUM_HIDDEN1];
    double z_h2[NUM_HIDDEN2], h2[NUM_HIDDEN2];
    double z_h3[NUM_HIDDEN3], h3[NUM_HIDDEN3];
    double z_h4[NUM_HIDDEN4], h4[NUM_HIDDEN4];
    double z_out[NUM_OUTPUTS], outputs[NUM_OUTPUTS];

    // He-init
    double scale;
    scale = sqrt(2.0 / NUM_FEATURES);
    for (int i = 0; i < NUM_HIDDEN1; i++) {
        b_h1[i] = 0.0;
        for (int j = 0; j < NUM_FEATURES; j++) W_input_h1[i][j] = rand_uniform() * scale;
    }
    scale = sqrt(2.0 / NUM_HIDDEN1);
    for (int i = 0; i < NUM_HIDDEN2; i++) {
        b_h2[i] = 0.0;
        for (int j = 0; j < NUM_HIDDEN1; j++) W_h1_h2[i][j] = rand_uniform() * scale;
    }
    scale = sqrt(2.0 / NUM_HIDDEN2);
    for (int i = 0; i < NUM_HIDDEN3; i++) {
        b_h3[i] = 0.0;
        for (int j = 0; j < NUM_HIDDEN2; j++) W_h2_h3[i][j] = rand_uniform() * scale;
    }
    scale = sqrt(2.0 / NUM_HIDDEN3);
    for (int i = 0; i < NUM_HIDDEN4; i++) {
        b_h4[i] = 0.0;
        for (int j = 0; j < NUM_HIDDEN3; j++) W_h3_h4[i][j] = rand_uniform() * scale;
    }
    scale = sqrt(2.0 / NUM_HIDDEN4);
    for (int i = 0; i < NUM_OUTPUTS; i++) {
        b_out[i] = 0.0;
        for (int j = 0; j < NUM_HIDDEN4; j++) W_h4_out[i][j] = rand_uniform() * scale;
    }

    int epochs = 10000;
    double learning_rate = 0.003;
    double val_check = 0.0;
    int times_no_improve = 0;

    for (int e = 0; e < epochs; e++) {
        double train_loss = 0.0;
        for (int i = 0; i < datasets.train.size; i++) {
            forward_propagation(datasets.train.X[i],
                W_input_h1, b_h1, W_h1_h2, b_h2, W_h2_h3, b_h3,
                W_h3_h4, b_h4, W_h4_out, b_out,
                h1, h2, h3, h4, outputs,
                z_h1, z_h2, z_h3, z_h4, z_out);

            double loss[NUM_OUTPUTS];
            mse_per_output(datasets.train.y[i], outputs, NUM_OUTPUTS, loss);
            train_loss += (loss[0] + loss[1]) / 2.0;

            back_propagation(datasets.train.X[i], datasets.train.y[i],
                W_input_h1, b_h1, W_h1_h2, b_h2, W_h2_h3, b_h3,
                W_h3_h4, b_h4, W_h4_out, b_out,
                h1, h2, h3, h4, outputs,
                z_h1, z_h2, z_h3, z_h4, z_out, learning_rate);
        }
        train_loss /= datasets.train.size;

        double val_loss = 0.0;
        for (int i = 0; i < datasets.val.size; i++) {
            forward_propagation(datasets.val.X[i],
                W_input_h1, b_h1, W_h1_h2, b_h2, W_h2_h3, b_h3,
                W_h3_h4, b_h4, W_h4_out, b_out,
                h1, h2, h3, h4, outputs,
                z_h1, z_h2, z_h3, z_h4, z_out);

            // Denormalize both prediction and ground truth
            double pred_denorm[NUM_OUTPUTS];
            double true_denorm[NUM_OUTPUTS];
            for (int j = 0; j < NUM_OUTPUTS; j++) {
                pred_denorm[j] = denorm_output(outputs[j], output_min[j], output_max[j]);
                true_denorm[j] = denorm_output(datasets.val.y[i][j], output_min[j], output_max[j]);
            }

            // Compute denormalized MSE
            double loss[NUM_OUTPUTS];
            mse_per_output(true_denorm, pred_denorm, NUM_OUTPUTS, loss);
            val_loss += (loss[0] + loss[1]) / 2.0;
        }
       val_loss /= datasets.val.size;
        // minska learning rate om ingen förbättring på 10 epoker

        if( val_check == 0.0 || val_loss < val_check ){
            val_check = val_loss;
            times_no_improve = 0;
        } else {
            times_no_improve++;
            if( times_no_improve >= 10 ){
                learning_rate *= 0.95;
                printf("No improvement, reducing learning rate to %f\n", learning_rate);
            }
        }
        if (e == 9000){
            learning_rate = 0.0001;
            printf("Learning rate satt till %f\n", learning_rate);
        }    

        printf("Epoch %d: Train Loss = %f, Val Loss = %.15e\n", e + 1, train_loss, val_loss);
        fprintf(val_log, "%d %.15e\n", e + 1, val_loss);
        fflush(val_log); // säkerställ att datan skrivs direkt till filen
    }

    // Test
    double test_loss[NUM_OUTPUTS] = {0.0, 0.0};
    for (int i = 0; i < datasets.test.size; i++) {
        forward_propagation(datasets.test.X[i],
            W_input_h1, b_h1, W_h1_h2, b_h2, W_h2_h3, b_h3,
            W_h3_h4, b_h4, W_h4_out, b_out,
            h1, h2, h3, h4, outputs,
            z_h1, z_h2, z_h3, z_h4, z_out);

        for (int j = 0; j < NUM_OUTPUTS; j++) {
        double pred_denorm = denorm_output(outputs[j], output_min[j], output_max[j]);
        double true_denorm = denorm_output(datasets.test.y[i][j], output_min[j], output_max[j]);
        double diff = pred_denorm - true_denorm;
        test_loss[j] += diff * diff;
    }
    }

    test_loss[0] /= datasets.test.size;
    printf("Test Loss (MSE) för output  compressor = %.10e\n", test_loss[0]);
    test_loss[1] /= datasets.test.size;
    printf("Test Loss (MSE) för output  turbine = %.10e\n", test_loss[1]);
    fclose(file);
    fclose(val_log);
    return 0;
}
