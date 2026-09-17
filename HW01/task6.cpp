#include <cstdio>
#include <cstdlib>
#include <iostream>

int main(int argc, char *argv[]) {
    int N = std::atoi(argv[1]);

    for (int i = 0; i <= N; i++) {
        printf("%d", i);
        if (i != N) {
            printf(" ");
        }
    }
    printf("\n");

    for (int i = N; i >= 0; i--) {
        std::cout << i;
        if (i != 0) {
            std::cout << " ";
        }
    }
    std::cout << std::endl;

    return 0;
}
