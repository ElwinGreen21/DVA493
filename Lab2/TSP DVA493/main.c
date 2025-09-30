#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define N 52              
#define POP_SIZE 500
#define GENERATIONS 50
#define TOURNAMENT 7
#define CROSSOVER_RATE 0.9
#define MUTATION_RATE 0.2
#define ELITE 5

typedef struct {
    int route[N - 1];   
    double dist;       
    double fitness;    
} Individual;

double coords[N][2];       
double distMatrix[N][N];   
Individual population[POP_SIZE];
Individual newPop[POP_SIZE];
Individual bestEver;


double euclidean(int i, int j) {
    double dx = coords[i][0] - coords[j][0];
    double dy = coords[i][1] - coords[j][1];
    return sqrt(dx * dx + dy * dy);
}

void computeDistMatrix() {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            distMatrix[i][j] = euclidean(i, j);
        }
    }
}


void evaluate(Individual* ind) {
    double length = 0.0;
    int prev = 0; 
    for (int i = 0; i < N - 1; i++) {
        int city = ind->route[i] - 1; 
        length += distMatrix[prev][city];
        prev = city;
    }
    length += distMatrix[prev][0]; 

    
    double penalty = (length > 8000) ? 10.0 * (length - 8000) : 0.0;

    ind->dist = length;
    ind->fitness = 1.0 / (length + penalty + 1e-6);
}


void randomRoute(int* route) {
    for (int i = 0; i < N - 1; i++) route[i] = i + 2; 
    for (int i = N - 2; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = route[i];
        route[i] = route[j];
        route[j] = tmp;
    }
}

void copyInd(Individual* dest, Individual* src) {
    for (int i = 0; i < N - 1; i++) {
        dest->route[i] = src->route[i];
    }
    dest->dist = src->dist;
    dest->fitness = src->fitness;
}


Individual* tournamentSelect() {
    Individual* best = &population[rand() % POP_SIZE];
    for (int i = 1; i < TOURNAMENT; i++) {
        Individual* challenger = &population[rand() % POP_SIZE];
        if (challenger->fitness > best->fitness)
            best = challenger;
    }
    return best;
}


void crossover(Individual* p1, Individual* p2, Individual* child) {
    //A mix of parents that creates a child
    int start = rand() % (N - 1);
    int end = rand() % (N - 1);

    if (start > end) { 
        int tmp = start; 
        start = end; 
        end = tmp; 
    }

    int used[N + 1] = { 0 };
    for (int i = start; i <= end; i++) {
        child->route[i] = p1->route[i];
        used[child->route[i]] = 1;
    }
    int idx = (end + 1) % (N - 1);
    int pos = (end + 1) % (N - 1);
    for (int k = 0; k < N - 1; k++) {
        int city = p2->route[idx];
        if (!used[city]) {
            child->route[pos] = city;
            pos = (pos + 1) % (N - 1);
            used[city] = 1;
        }
        idx = (idx + 1) % (N - 1);
    }
}

void mutate2opt(Individual* ind) {
    int i = rand() % (N - 2);
    int j = i + 1 + rand() % (N - 2 - i);

    while (i < j) {
        int tmp = ind->route[i];
        ind->route[i] = ind->route[j];
        ind->route[j] = tmp;
        i++;
        j--;
    }
}

void local2opt(Individual* ind) {
    int improved = 1;
    while (improved) {
        improved = 0;
        for (int i = 0; i < N - 3; i++) {
            for (int j = i + 2; j < N - 1; j++) {
                
                int a = (i == 0) ? 0 : ind->route[i - 1] - 1;
                int b = ind->route[i] - 1;
                int c = ind->route[j] - 1;
                int d = (j == N - 2) ? 0 : ind->route[j + 1] - 1;

                double oldDist = distMatrix[a][b] + distMatrix[c][d];
                double newDist = distMatrix[a][c] + distMatrix[b][d];

                if (newDist < oldDist) {
                    // turn segment [i..j]
                    int left = i, right = j;
                    while (left < right) {
                        int tmp = ind->route[left];
                        ind->route[left] = ind->route[right];
                        ind->route[right] = tmp;
                        left++; right--;
                    }
                    improved = 1;
                }
            }
        }
    }
    evaluate(ind);
}


void initPopulation() {
    for (int i = 0; i < POP_SIZE; i++) {
        randomRoute(population[i].route);
        evaluate(&population[i]);
        if (i == 0 || population[i].dist < bestEver.dist)
            copyInd(&bestEver, &population[i]);
    }
}

void evolve() {
    for (int gen = 0; gen < GENERATIONS; gen++) {
        
        for (int e = 0; e < ELITE; e++) {
            int bestIndex = 0;
            for (int i = 1; i < POP_SIZE; i++) {
                if (population[i].dist < population[bestIndex].dist)
                    bestIndex = i;
            }
            copyInd(&newPop[e], &population[bestIndex]);
        }

        
        for (int i = ELITE; i < POP_SIZE; i++) {
            Individual* p1 = tournamentSelect();
            Individual* p2 = tournamentSelect();
            Individual child;

            if (((double)rand() / RAND_MAX) < CROSSOVER_RATE)
                crossover(p1, p2, &child);
            else
                copyInd(&child, p1);

            if (((double)rand() / RAND_MAX) < MUTATION_RATE)
                mutate2opt(&child);
            
            
            local2opt(&child);


            evaluate(&child);
            copyInd(&newPop[i], &child);
        }

        for (int i = 0; i < POP_SIZE; i++) {
            copyInd(&population[i], &newPop[i]);
            if (population[i].dist < bestEver.dist)
                copyInd(&bestEver, &population[i]);
        }

        
         printf("Generation %d: Best = %.2f\n", gen+1, bestEver.dist);
        
    }
}

int main() {
    srand(time(NULL));

    FILE* fp = fopen("berlin52.tsp", "r");
    if (!fp) { printf("couldn't open berlin52.tsp\n"); return 1; }

    char line[256];
    while (fgets(line, sizeof(line), fp)) {
        if (strncmp(line, "NODE_COORD_SECTION", 18) == 0) {
            break;
        }
    }
    for (int i = 0; i < N; i++) {
        int id; double x, y;
        fscanf_s(fp, "%d %lf %lf", &id, &x, &y);
        coords[i][0] = x;
        coords[i][1] = y;
    }
    fclose(fp);

    computeDistMatrix();
    initPopulation();
    evolve();

    printf("\nBest route found (distance %.2f):\n", bestEver.dist);
    printf("1 ");
    for (int i = 0; i < N - 1; i++) printf("%d ", bestEver.route[i]);
    printf("1\n");

    return 0;
}
