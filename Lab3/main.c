//Assignment 3: Unsupervised Learning
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define ROWS 178
#define COLS 13

int file_reader(int classes[ROWS], double data[ROWS][COLS]) {
    FILE *fp;
    int i, j;

    fp = fopen("WINE.txt", "r");
    if (fp == NULL) {
        perror("Kunde inte öppna filen");
        return 1;
    }

    for (i = 0; i < ROWS; i++) {
        // Läs först klasskolumnen
        if (fscanf(fp, "%d", &classes[i]) != 1) {
            fprintf(stderr, "Fel vid läsning av klass på rad %d\n", i+1);
            return 1;
        }
        // Läs sedan features
        for (j = 0; j < COLS; j++) {
            if (fscanf(fp, "%lf", &data[i][j]) != 1) {
                fprintf(stderr, "Fel vid läsning av feature på rad %d kolumn %d\n", i+1, j+1);
                return 1;
            }
        }
    }

    fclose(fp);
    return 0;
}

int normilize_data(double data[ROWS][COLS]) {
    int j = 0;
    int i = 0;
    double sum = 0.0;
    double mean = 0.0;
    
    for(j = 0; j < COLS; j++){
        sum = 0.0;
        mean = 0.0;
         
        for (i = 0; i < ROWS; i++) {
            sum += data[i][j];
        }
        mean = sum / ROWS;
        for (i = 0; i < ROWS; i++) {
            data[i][j] -= mean;
        }
    }

    return 0;
}

compute_covariance(double data[ROWS][COLS], double cov[COLS][COLS]) {
    int i = 0;
    int j = 0;
    int k = 0;

    for (i = 0; i < COLS; i++) {
        for (j = 0; j < COLS; j++) {
            cov[i][j] = 0.0;
            
            for (k = 0; k < ROWS; k++) {
                cov[i][j] += data[k][i] * data[k][j];
            }
            cov[i][j] /= (ROWS - 1); 
        }
    }

    return 0;
}


int main() {
    

    // En kolumn för klass (int) och en matris för features (double)
    int classes[ROWS];
    double data[ROWS][COLS]; 
    double cov[COLS][COLS];

    file_reader(classes, data);
    normilize_data(data);
    
    ///*
    // Testutskrift: första raden
    printf("Klass: %d\n", classes[0]);
    printf("Features: ");
    for (int j = 0; j < COLS - 1; j++) {
        printf("%lf ", data[0][j]);
    }
    printf("\n");

    return 0;
    //*/
}
