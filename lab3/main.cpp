#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <chrono>
#include <mpi.h>

using namespace std;
using namespace chrono;
using Matrix = vector<vector<int>>;

bool readMatrix(const string& filename, Matrix& mat) {
    ifstream fin(filename);
    if (!fin) return false;
    string line;
    while (getline(fin, line)) {
        istringstream iss(line);
        vector<int> row;
        int val;
        while (iss >> val) row.push_back(val);
        mat.push_back(row);
    }
    return !mat.empty();
}

void writeMatrix(const string& filename, const Matrix& mat) {
    ofstream fout(filename);
    for (auto& row : mat) {
        for (int val : row) fout << val << " ";
        fout << "\n";
    }
}

void multiplyAndMeasure(const string& fileA, const string& fileB, const string& resultFile, int rank, int size) {
    Matrix A, B;
    int aRows = 0, aCols = 0, bCols = 0;

    if (rank == 0) {
        readMatrix(fileA, A);
        readMatrix(fileB, B);
        aRows = A.size(); aCols = A[0].size(); bCols = B[0].size();
    }

    MPI_Bcast(&aRows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&aCols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&bCols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) B.resize(aCols, vector<int>(bCols));
    for (int i = 0; i < aCols; ++i)
        MPI_Bcast(B[i].data(), bCols, MPI_INT, 0, MPI_COMM_WORLD);

    int localStart = rank * aRows / size;
    int localEnd = (rank + 1) * aRows / size;
    int localRows = localEnd - localStart;

    Matrix localA(localRows, vector<int>(aCols));
    if (rank == 0) {
        for (int i = 1; i < size; ++i)
            for (int r = i * aRows / size; r < (i + 1) * aRows / size; ++r)
                MPI_Send(A[r].data(), aCols, MPI_INT, i, 0, MPI_COMM_WORLD);
        for (int r = 0; r < localRows; ++r)
            localA[r] = A[r];
    }
    else {
        for (int r = 0; r < localRows; ++r)
            MPI_Recv(localA[r].data(), aCols, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    high_resolution_clock::time_point start, end;
    if (rank == 0) start = high_resolution_clock::now();

    Matrix localC(localRows, vector<int>(bCols, 0));
    for (int i = 0; i < localRows; ++i)
        for (int j = 0; j < bCols; ++j)
            for (int k = 0; k < aCols; ++k)
                localC[i][j] += localA[i][k] * B[k][j];

    Matrix C;
    if (rank == 0) C.resize(aRows, vector<int>(bCols));

    for (int i = 0; i < localRows; ++i)
        if (rank == 0) C[i] = localC[i];
        else MPI_Send(localC[i].data(), bCols, MPI_INT, 0, 1, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int p = 1; p < size; ++p) {
            int startRow = p * aRows / size;
            int endRow = (p + 1) * aRows / size;
            for (int i = startRow; i < endRow; ++i)
                MPI_Recv(C[i].data(), bCols, MPI_INT, p, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        end = high_resolution_clock::now();
        duration<double> duration = end - start;

        writeMatrix(resultFile, C);
        cout << "Matrix size: " << aRows << " x " << aRows << " | Time: " << duration.count() << " seconds\n\n";
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    vector<tuple<string, string, string>> tasks = {
        {"../Matrix/matrixA.txt", "../Matrix/matrixB.txt", "../Matrix/result.txt"},
        {"../Matrix/matrixA_10k.txt", "../Matrix/matrixB_10k.txt", "../Matrix/result_10k.txt"},
        {"../Matrix/matrixA_250k.txt", "../Matrix/matrixB_250k.txt", "../Matrix/result_250k.txt"},
        {"../Matrix/matrixA_1kk.txt", "../Matrix/matrixB_1kk.txt", "../Matrix/result_1kk.txt"},
        {"../Matrix/matrixA_4kk.txt", "../Matrix/matrixB_4kk.txt", "../Matrix/result_4kk.txt"}
    };

    for (auto& [a, b, result] : tasks) {
        MPI_Barrier(MPI_COMM_WORLD);
        multiplyAndMeasure(a, b, result, rank, size);
    }

    MPI_Finalize();
    return 0;
}
