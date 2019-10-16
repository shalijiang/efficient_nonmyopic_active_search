#include <iostream>
#include <random>
#include <cstdlib>
using namespace std;
int main(int argc, char *argv[]){
    default_random_engine generator;
    uniform_real_distribution<double> uniform(0.0, 1.0);
    int n = atoi(argv[1]);
    for (int i = 0; i < n; i++){
        double num = uniform(generator);
        cout << num << endl;
    }
    return 0;
}
