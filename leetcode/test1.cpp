#include <iostream>
#include <cstdlib>
#include<ctime>
using namespace std;
 
int main()
{
    srand((unsigned) time(NULL));
    for (int i = 0; i < 5; i++)
            cout << rand() << endl;
}