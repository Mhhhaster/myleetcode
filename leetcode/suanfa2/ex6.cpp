#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>

int val[100000],ptr[100000];
int head;

int Search(int x,int i){
    int count=1;
    while(x>val[i]){
        count++;
        i=ptr[i];
    }
    return count;
}

int A(int x){
    return Search(x,head);
}

int B(int x){
    int i=head;
    int max=val[i];
    int y=0;
    for(int j=0;j<sqrt(n);j++){
        y=val[j];
        if((max<y)&&(y<=x)){
            i=j;
            max=y;
        }
    }
    return Search(x,i)+(int)sqrt(n);
}

int C(int x){
    int i=rand()%n;
    int max=val[i];
    int temp=0;
    int y=0;
    for(int j=0;j<sqrt(n);j++){
        temp=rand()%n;
        y=val[temp];
        if((max<y)&&(y<=x)){
            i=temp;
            max=y;
        }
    }
    return Search(x,i)+(int)sqrt(n);
}

int D(int x){
    int i=rand()%n;
    int y=val[i];
    if(x<y){
        return Search(x,head)+1;
    }
    else{
        if(x>y)
            return Search(x,ptr[i])+1;
        else
            return 1;
    }
}