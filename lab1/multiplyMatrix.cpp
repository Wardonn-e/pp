#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <chrono>


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
        // строковый поток для разбора чисел из строки line
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
    for (int i = 0; i < aRows; ++i)
        for (int j = 0; j < bCols; ++j)
            for (int k = 0; k < aCols; ++k)
                C[i][j] += A[i][k] * B[k][j];
    return C;
}

int main() {
    Matrix A, B, C;

    if (!readMatrix("../Matrix/matrixA.txt", A)) return 1;
    if (!readMatrix("../Matrix/matrixB.txt", B)) return 1;

    int aRows = A.size(), aCols = A[0].size();
    int bRows = B.size(), bCols = B[0].size();

    if (aCols != bRows) {
        cerr << "Error: incompatible matrix dimensions for multiplication." << endl;
        return 1;
    }

    auto start = high_resolution_clock::now();
    C = multiplyMatrices(A, B);
    auto end = high_resolution_clock::now();

    duration<double> duration = end - start;
    writeMatrix("../Matrix/resultMatrix.txt", C);

    cout << "time: " << duration.count() << " seconds." << endl;
    cout << "size: " << aRows << "x" << aCols << " * " << bRows << "x" << bCols
        << " = " << aRows << "x" << bCols << " result matrix." << endl;

    return 0;
}
