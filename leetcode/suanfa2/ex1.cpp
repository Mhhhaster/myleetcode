#include<iostream>
#include<ctime>  //需要用到系统时间
#include<cstdlib>
#include<math.h>
using namespace std;

int main(){
    int i, in_circle, n;
    double x, y, tmp, pi;
    while(1){
        cout<<"Input a number: ";
        cin>>n;
        srand((unsigned) time(NULL)); //利用系统时间设置随机种子，每次运行只需设置一次
        in_circle=0;
        for(i=0;i<n;i++){
            x=rand()/(double)(RAND_MAX);
            tmp=pow(x,2)+pow(x,2);
            if(tmp<=1)
                in_circle++;
        }
        pi=(4*in_circle)/(double)n;
        cout.precision(6);
        cout.setf(ios::fixed);
        cout<<"pi is: "<<pi<<endl;
    }
    return 0;
}