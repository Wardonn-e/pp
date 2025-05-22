#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace chrono;

typedef vector<vector<int>> Matrix;

bool readMatrix(const string& filename, Matrix& mat) {
    ifstream fin(filename);
    if (!fin) {
        cerr << "Error opening file: " << filename << endl;
        return false;
    }
    string line;
    while (getline(fin, line)) {
        istringstream iss(line);
        vector<int> row;
        int num;
        while (iss >> num)
            row.push_back(num);
        if (!row.empty())
            mat.push_back(row);
    }
    return !mat.empty();
}

void writeMatrix(const string& filename, const Matrix& mat) {
    ofstream fout(filename);
    for (const auto& row : mat) {
        for (int val : row)
            fout << val << " ";
        fout << "\n";
    }
}

Matrix multiplyMatrices(const Matrix& A, const Matrix& B) {
    int aRows = A.size(), aCols = A[0].size();
    int bCols = B[0].size();
    Matrix C(aRows, vector<int>(bCols, 0));

#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < aRows; ++i) {
        for (int j = 0; j < bCols; ++j) {
            int sum = 0;
            for (int k = 0; k < aCols; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
    return C;
}


int main() {
    Matrix A, B;

    if (!readMatrix("../Matrix/matrixA_4kk.txt", A)) return 1;
    if (!readMatrix("../Matrix/matrixB_4kk.txt", B)) return 1;

    int aRows = A.size(), aCols = A[0].size();
    int bRows = B.size(), bCols = B[0].size();

    if (aCols != bRows) {
        cerr << "Error: incompatible matrix dimensions for multiplication." << endl;
        return 1;
    }

    vector<int> threadCounts = { 28 };

    cout << "Matrix size: " << aRows << "x" << aCols << " * " << bRows << "x" << bCols
        << " = " << aRows << "x" << bCols << " result\n" << endl;

    for (int threads : threadCounts) {
        omp_set_num_threads(threads);

        auto start = high_resolution_clock::now();
        Matrix C = multiplyMatrices(A, B);
        auto end = high_resolution_clock::now();

        duration<double> duration = end - start;

        cout << "Threads: " << threads << " | Time: " << duration.count() << " seconds" << endl;
    }

    return 0;
}
