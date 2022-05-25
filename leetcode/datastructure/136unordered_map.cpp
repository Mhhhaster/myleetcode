class Solution {
public:
    int singleNumber(vector<int>& nums) {
        unordered_map<int,bool>once;
        for(int i=0;i<nums.size();i++){
            if(once[nums[i]]==false)
                once[nums[i]]=true;
            else
                once.erase(once.find(nums[i]));
        }
        return once.begin()->first;
    }
};