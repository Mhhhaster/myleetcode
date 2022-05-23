#include<iostream>
#include<cstdlib>
#include<math.h>
#include<time.h>
using namespace std;

bool Btest(int a,int n){
    int s=0;
    int t=n-1;
    int x=1;
    while(t%2!=1){
        t/=2;
        s++;
    }
    for(int i=0;i<t;i++){
        x*=a;
        x%=n;
    }
    if((x==1)||(x==n-1))
        return true;
    for(int i=1;i<=s-1;i++){
        x=(x*x)%n;
        if(x==n-1)
            return true;
    }
    return false;
}

bool MillRab(int n){
    int a=0;
    srand((unsigned) time(NULL));
    a=rand()%(n-3)+2;
    return Btest(a,n);
}

bool RepeatMillRab(int n,int k){
    for(int i=1;i<=k;i++){
        if(!MillRab(n))
            return false;
    }
    return true;
}

int CerAlgorithm(int n){
    int x=0;
    int count=1;
    bool flag=true;
    for(int j=3;j<=n;j+=2){
        x=(int)sqrt(j);
        for(int i=2;i<=x;i++){
            if(j%i==0){
                flag=false;
                break;
            }
        }
        if(flag)
            count++;
        flag=true;
    }
    return count;
}

int PrintPrimes(int n,int odd_list[]){
    odd_list[0]=2;
    odd_list[1]=3;
    int count=2;
    int odd=5;
    int x=0;
    while(odd<=n){
        if(RepeatMillRab(odd,(int)(log((double)odd)/log(10.0)))){
            odd_list[count++]=odd;
        }
        odd+=2;
    }
    return count;
}

int main(){
    int odd_list[1250];
    cout<<"确定算法计算出100-10000内有"<<CerAlgorithm(10000)-CerAlgorithm(100)<<"个素数"<<endl<<endl;
    int sum_10000=PrintPrimes(10000,odd_list);
    int sum_100=PrintPrimes(100,odd_list);
    int count=1;
    cout<<"概率算法计算出100-10000内的素数打印如下： "<<endl;
    for(int i=sum_100;i<sum_10000;i++){
        cout<<odd_list[i]<<" ";
        count++;
        if(count%10==0)
            cout<<endl;
    }
    cout<<"概率算法计算出100-10000内有"<<sum_10000-sum_100<<"个素数"<<endl<<endl<<endl;
    return 0;
}