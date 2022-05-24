/*my code*/
class Solution {
public:
	vector<int> twoSum(vector<int>& nums, int target) {
		unordered_map<int, int>M;
		for (int i = 0; i < nums.size(); i++) {
			if (M[target - nums[i]] != 0)
				return { M[target - nums[i]] - 1,i };
			M[nums[i]] = i + 1;
		}
		return {};
	}
};

/*stardard answer*/
class Solution {
public:
	vector<int> twoSum(vector<int>& nums, int target) {
		unordered_map<int, int> hashtable;
		for (int i = 0; i < nums.size(); ++i) {//注意到这里是++i
			auto it = hashtable.find(target - nums[i]);
			if (it != hashtable.end()) {
				return { it->second, i };
			}
			hashtable[nums[i]] = i;
		}
		return {};
	}
};

/*learn:
都采用unordered_map实现对实数范围内的索引，大范围依次映射到小范围
我通过哈希表索引结果是否为0来判断该元素是否被加入了，而标准答案通过find查询*/