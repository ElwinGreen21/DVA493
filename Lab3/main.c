//Assignment 3: Unsupervised Learning
#include <stdio.h>
#include <stdlib.h>

#define ROWS 178
#define COLS 14

int main() {
    FILE *fp;
    int i, j;

    // En kolumn för klass (int) och en matris för features (double)
    int classes[ROWS];
    double data[ROWS][COLS - 1]; // 13 features

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
        for (j = 0; j < COLS - 1; j++) {
            if (fscanf(fp, "%lf", &data[i][j]) != 1) {
                fprintf(stderr, "Fel vid läsning av feature på rad %d kolumn %d\n", i+1, j+1);
                return 1;
            }
        }
    }

    fclose(fp);

    // Testutskrift: första raden
    printf("Klass: %d\n", classes[0]);
    printf("Features: ");
    for (j = 0; j < COLS - 1; j++) {
        printf("%lf ", data[0][j]);
    }
    printf("\n");

    return 0;
}
