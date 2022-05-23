#include<iostream>
#include<ctime>
#include<cstdlib>
#include<math.h>
using namespace std;

double uniform(double a, double b){
    double x=rand()/(double)(RAND_MAX);
    return (b-a)*x+a;
}

double HitorMiss(int n){
    int k=0;
    double x,y;
    for(int i=0;i<n;i++){
        x=uniform(0,1);
        y=uniform(0,1);
        if(y<=sqrt(1-x*x))
            k++;
    }
    return (double)k/n;
}

int main(){
    int n=0;
    double pi;
    while(1){
        cout<<"input a number: ";
        cin>>n;
        pi=4*HitorMiss(n);
        cout.precision(6);
        cout.setf(ios::fixed);
        cout<<"pi is: "<<pi<<endl;
    }
    return 0;
}