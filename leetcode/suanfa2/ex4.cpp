#include<iostream>
#include<ctime>
#include<cstdlib>
#include<math.h>
#define PI 3.1415926
using namespace std;

int is_inArray(int tmp, int *array,int length){
    for(int i=0;i<length;i++){
        if(array[i]==tmp)
            return 1;
    }
    return 0;
}

int SetCount(int n){
    int k=0;
    int tmp,length=0;
    int *array=(int *)malloc(sizeof(int)*n);
    tmp=(rand()*rand()*rand())%n+1;
    do{
        k++;
        array[length++]=tmp;
        tmp=(rand()*rand()*rand())%n+1;
    }while(!is_inArray(tmp,array,length));
    free(array);
    return (int)2*k*k/PI;
}

int main(){
    int n=0;
    int num=100;
    int n_sum=0;
    while(1){
        n_sum=0;
        cout<<"input a number: ";
        cin>>n;
        srand((unsigned) time(NULL));
        for(int i=0;i<num;i++)
            n_sum+=SetCount(n);
        cout<<"number of elements is: "<<n_sum/num<<endl;
    }
    return 0;
}