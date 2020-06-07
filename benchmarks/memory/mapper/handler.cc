#include <iostream>
#include <chrono>

using namespace std;

static const size_t N_TRIALS = 1024 * 512;

static const size_t ARRAY_SIZE = 1024 * 1024 * 2;
static double a[ARRAY_SIZE];
static double b[ARRAY_SIZE];
static double c[ARRAY_SIZE];

int main() {
    for (size_t j = 0; j < ARRAY_SIZE; j++) {
        a[j] = 1.0;
        b[j] = 2.0;
        c[j] = 3.0;
    }
    double scalar = 4.0;

    auto start = chrono::steady_clock::now();
    for (size_t i = 0; i < N_TRIALS; i++) {
        for (size_t j = 0; j < ARRAY_SIZE; j++) {
            a[j] = b[j] + c[j] * scalar;
        }
    }
    auto end = chrono::steady_clock::now();
    chrono::duration<double> elapsed = end - start;

    cout << elapsed.count() << endl;
    return 0;
}
