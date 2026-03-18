#include <iostream>
#include <sstream>
#include <stdio.h>

using namespace std;

char buf[1000000000];

int main()
{
    int n = 0;
    for (int i = 1; i <= 5; ++i) {
        stringstream ss;
        ss << "data_batch_" << i << ".bin";
        FILE *filbin = fopen(ss.str().c_str(), "rb");
        n += (int)fread(buf + n, 1, 1000000000 - n, filbin);
        fclose(filbin);
    }
    FILE *filbin = fopen("test_batch.bin", "rb");
    n += (int)fread(buf + n, 1, 1000000000 - n, filbin);
    fclose(filbin);
    int i = 0;
    char *pc = buf;
    FILE *filbinout = fopen("CIFAR10.bin", "wb");
    FILE *filtargetout = fopen("CIFAR10.target.txt", "wt");
    const int siz = 32;
    const int nchannels = 3;
    const int m = siz * siz * nchannels;
//    fwrite(&siz, sizeof(int), 1, filbinout);
//    fwrite(&siz, sizeof(int), 1, filbinout);
//    fwrite(&nchannels, sizeof(int), 1, filbinout);
    while (i < n) {
        fprintf(filtargetout, "%d\n", (int)*pc++);
        fwrite(pc, 1, m, filbinout);
        pc += m;
        i += 1 + m;
    }
    _fcloseall();
    return 0;
}
