#include<iostream>
#include<ctime>
#include<cstdlib>
#include<math.h>
using namespace std;

double uniform(double a, double b){
    double x=rand()/(double)(RAND_MAX);
    return (b-a)*x+a;
}

double f(double x){
    return x;
}

double HitorMiss(double a,double b,double c,double d,int n,double (*f)(double)){
    int k=0;
    double x,y;
    for(int i=0;i<n;i++){
        x=uniform(a,b);
        y=uniform(a,b);
        if(y<=f(x))
            k++;
    }
    return (double)k/n*(b-a)*(d-c)+c*(b-a);
}

int main(){
    int n;
    double a,b,c,d;
    while(1){
        cout<<"input the numbers: ";
        cin>>a>>b>>n;
        c=f(a);
        d=f(b);
        srand((unsigned) time(NULL));
        cout.precision(6);
        cout.setf(ios::fixed);
        cout<<"integration is: "<<HitorMiss(a,b,c,d,n,f)<<endl;
    }
    return 0;
}