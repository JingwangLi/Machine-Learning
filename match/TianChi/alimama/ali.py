import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss


def timestamp_datetime(value):
    format = '%Y-%m-%d %H:%M:%S'
    value = time.localtime(value)
    dt = time.strftime(format, value)
    return dt

def convert_train(data):
    data.drop_duplicates(inplace=True)
    data['time'] = data.context_timestamp.apply(timestamp_datetime)
    data['day'] = data.time.apply(lambda x: int(x[8:10]))
    data['hour'] = data.time.apply(lambda x: int(x[11:13]))
    user_query_day = data.groupby(['user_id', 'day']).size(
    ).reset_index().rename(columns={0: 'user_query_day'})
    data = pd.merge(data, user_query_day, 'left', on=['user_id', 'day'])
    user_query_day_hour = data.groupby(['user_id', 'day', 'hour']).size().reset_index().rename(
        columns={0: 'user_query_day_hour'})
    data = pd.merge(data, user_query_day_hour, 'left',
                    on=['user_id', 'day', 'hour'])

    data['user_gender_id'] = data['user_gender_id'].replace(-1,0)
    data = data.replace(-1,np.nan)
    data = data.dropna(axis = 0)

    user_item_trade = data[data['is_trade'] == 1].groupby(['user_id','item_id','context_timestamp'])['context_timestamp'].size()
    def num_user_buy_item(data):
         x = data['user_id']
         y = data['item_id']
         time = data['context_timestamp']
         try:
             trades = user_item_trade[(x,y,)]
             return trades[trades.index < time].size
         except:
             return 0
    data['num_user_buy_item'] = data.apply(num_user_buy_item,axis = 1)

    item_category_property = data[['item_id','item_category_list','item_property_list']].drop_duplicates()
    item_category_property['item_category'] = item_category_property['item_category_list'].apply(lambda x:x.split(';'))
    item_category_property['item_property'] = item_category_property['item_property_list'].apply(lambda x:x.split(';'))

    data =pd.merge(data,item_category_property,'left',on = 'item_id')

    data['category_property'] = data['predict_category_property'].apply(lambda x:[i.split(':') for i in x.split(';')])
    data['predict_category'] = data['category_property'].apply(lambda x:[i[0] for i in x])
    data['predict_property'] = data['category_property'].apply(lambda x:[i[1] for i in x if len(i) > 1])

    def predict_category_match(x):
        real = x['item_category']
        pre = x['predict_category']
        cnt = 0
        for i in real:
            if i in pre:cnt += 1
        return  cnt

    def predict_property_match(x):
        real = x['item_property']
        pre = x['predict_property']
        cnt = 0
        for i in real:
            if i in pre:cnt += 1
        return  cnt

    data['category_match'] = data.apply(predict_category_match,axis=1)
    data['property_match'] = data.apply(predict_property_match,axis=1)

    data['buy_show_rate'] = data['item_sales_level']/data['item_pv_level']

    data['user_age_level'] = data['user_age_level'] - 1000
    item_user_mean_age_level = data[data['is_trade'] == 1].groupby('item_id')['user_age_level'].mean().reset_index().rename(
            columns = {'user_age_level':'item_user_mean_age_level'})
    data = pd.merge(data,item_user_mean_age_level,'left',on = 'item_id')
    data['item_user_mean_age_level'] = data['item_user_mean_age_level'].fillna(data['user_age_level'].mean())

    data['user_star_level'] = data['user_star_level'] - 3000
    item_user_mean_star_level = data[data['is_trade'] == 1].groupby('item_id')['user_star_level'].mean().reset_index().rename(
            columns = {'user_star_level':'item_user_mean_star_level'})
    data = pd.merge(data,item_user_mean_star_level,'left',on = 'item_id')
    data['item_user_mean_star_level'] = data['item_user_mean_star_level'].fillna(data['user_star_level'].mean())

    item_user_gender_mode = data[data['is_trade'] == 1].groupby('item_id')['user_gender_id'].agg(lambda x: x.value_counts().index[0]).reset_index().rename(
            columns = {'user_gender_id':'item_user_gender_mode'})
    data = pd.merge(data,item_user_gender_mode,'left',on = 'item_id')
    data['item_user_gender_mode'] = data['item_user_gender_mode'].fillna(data['user_gender_id'].agg(lambda x: x.value_counts().index[0]))

    data['user_occupation_id'] = data['user_occupation_id'] - 2000
    item_user_occupation_mode = data[data['is_trade'] == 1].groupby('item_id')['user_occupation_id'].agg(lambda x: x.value_counts().index[0]).reset_index().rename(
            columns = {'user_occupation_id':'item_user_occupation_mode'})
    data = pd.merge(data,item_user_occupation_mode,'left',on = 'item_id')
    data['item_user_occupation_mode'] = data['item_user_occupation_mode'].fillna(data['user_occupation_id'].agg(lambda x: x.value_counts().index[0]))
    data.to_csv('data_2.csv')
    return data



def convert_test(data,train):
    data['time'] = data.context_timestamp.apply(timestamp_datetime)
    data['day'] = data.time.apply(lambda x: int(x[8:10]))
    data['hour'] = data.time.apply(lambda x: int(x[11:13]))
    user_query_day = data.groupby(['user_id', 'day']).size(
    ).reset_index().rename(columns={0: 'user_query_day'})
    data = pd.merge(data, user_query_day, 'left', on=['user_id', 'day'])
    user_query_day_hour = data.groupby(['user_id', 'day', 'hour']).size().reset_index().rename(
        columns={0: 'user_query_day_hour'})
    data = pd.merge(data, user_query_day_hour, 'left',
                    on=['user_id', 'day', 'hour'])

    data = pd.merge(data,train[['user_id','item_id','num_user_buy_item']],'left',on = ['user_id','item_id'])
    data['num_user_buy_item'] = data['num_user_buy_item'].fillna(0)

    item_category_property = data[['item_id','item_category_list','item_property_list']].drop_duplicates()
    item_category_property['item_category'] = item_category_property['item_category_list'].apply(lambda x:x.split(';'))
    item_category_property['item_property'] = item_category_property['item_property_list'].apply(lambda x:x.split(';'))

    data =pd.merge(data,item_category_property,'left',on = 'item_id')
    data['category_property'] = data['predict_category_property'].apply(lambda x:[i.split(':') for i in x.split(';')])
    data['predict_category'] = data['category_property'].apply(lambda x:[i[0] for i in x])
    data['predict_property'] = data['category_property'].apply(lambda x:[i[1] for i in x if len(i) > 1])

    def predict_category_match(x):
        real = x['item_category']
        pre = x['predict_category']
        cnt = 0
        for i in real:
            if i in pre:cnt += 1
        return  cnt

    def predict_property_match(x):
        real = x['item_property']
        pre = x['predict_property']
        cnt = 0
        for i in real:
            if i in pre:cnt += 1
        return  cnt

    data['category_match'] = data.apply(predict_category_match,axis=1)
    data['property_match'] = data.apply(predict_property_match,axis=1)

    data['buy_show_rate'] = data['item_sales_level']/data['item_pv_level']
    print('sdfg')

    data['user_age_level'] = data['user_age_level'] - 1000
    data = pd.merge(data,train[['item_id','item_user_mean_age_level']].drop_duplicates(),'left',on = 'item_id')
    data['item_user_mean_age_level'] = data['item_user_mean_age_level'].fillna(train['user_age_level'].mean())

    data['user_star_level'] = data['user_star_level'] - 3000
    data = pd.merge(data,train[['item_id','item_user_mean_star_level']].drop_duplicates(),'left',on = 'item_id')
    data['item_user_mean_star_level'] = data['item_user_mean_star_level'].fillna(train['user_star_level'].mean())

    data = pd.merge(data,train[['item_id','item_user_gender_mode']].drop_duplicates(),'left',on = 'item_id')
    data['item_user_gender_mode'] = data['item_user_gender_mode'].fillna(train['user_gender_id'].agg(lambda x: x.value_counts().index[0]))

    data['user_occupation_id'] = data['user_occupation_id'] - 2000
    data = pd.merge(data,train[['item_id','item_user_occupation_mode']].drop_duplicates(),'left',on = 'item_id')
    data['item_user_occupation_mode'] = data['item_user_occupation_mode'].fillna(train['user_occupation_id'].agg(lambda x: x.value_counts().index[0]))

    return data


#ori_data = pd.read_csv('round1_ijcai_18_train_20180301.txt', sep=' ')
#ori_data.drop_duplicates(inplace=True)
#data = convert_data(ori_data)
if __name__ == "__main__":
    online = True# 这里用来标记是 线下验证 还是 在线提交
#    data = pd.read_csv('round1_ijcai_18_train_20180301.txt', sep=' ')
#    data = convert_train(data)


    if online == False:
        train = data.loc[data.day < 24]  # 18,19,20,21,22,23,24

    elif online == True:
        train = data.copy()
        test = pd.read_csv('round1_ijcai_18_test_a_20180301.txt', sep=' ')
        test = convert_test(test,train)

    features = ['num_user_buy_item','item_user_occupation_mode','item_user_gender_mode',
                'item_user_mean_star_level','item_user_mean_age_level','buy_show_rate','property_match','category_match',
                'item_id', 'item_brand_id', 'item_city_id', 'item_price_level', 'item_sales_level',
                'item_collected_level', 'item_pv_level', 'user_gender_id', 'user_occupation_id',
                'user_age_level', 'user_star_level', 'user_query_day', 'user_query_day_hour',
                'hour', 'shop_id', 'shop_review_num_level', 'shop_star_level',
                'shop_review_positive_rate', 'shop_score_service', 'shop_score_delivery', 'shop_score_description',
                ]
    target = ['is_trade']

    if online == False:
        clf = lgb.LGBMClassifier(num_leaves=63, max_depth=7, n_estimators=80, n_jobs=20)
        clf.fit(train[features], train[target], feature_name=features,
                categorical_feature=['user_gender_id', ])
        test['lgb_predict'] = clf.predict_proba(test[features],)[:, 1]
        print(log_loss(test[target], test['lgb_predict']))
    else:
        clf = lgb.LGBMClassifier(num_leaves=63, max_depth=7, n_estimators=80, n_jobs=20)
        clf.fit(train[features], train[target],
                categorical_feature=['user_gender_id', ])
        test['predicted_score'] = clf.predict_proba(test[features])[:, 1]
        test[['instance_id', 'predicted_score']].to_csv('baseline.csv', index=False,sep=' ')#保存在线提交结果