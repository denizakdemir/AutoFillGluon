import pandas as pd

def transform_data(df):
    # Filter based on given conditions
    df = df[(df['numhires8_1'] == 8) & 
            (df['related'] == 0) & 
            (df['tag_consent'] == 1) & 
            (df['tag_noembargo'] == 1)]
    
    # Additional transformations
    df['dpberr'] = (1 - (df['dpb1permissive'] + df['dpb1nonpermissive'])) > 0.5
    
    # Filter out based on conditions
    df = df[df['dpberr'] <= 0.5]
    
    # Additional transformations
    df.loc[df['disease'] == 200, 'disease'] = pd.NA
    
    # Rename columns
    df.rename(columns={
        'mmdist': 'myedist',
        'mmissdx': 'myeissdx',
        'mmissdsdx': 'myeissdsdx',
        'mmcytorisk': 'ted_cytorisk',
        'medhinc_cy': 'median_income',
        'urdbmpbdage_up': 'urdbmpbdage',
        'dparity1_up': 'dparity1',
        'drabomatch_dots': 'drabomatch'
    }, inplace=True)

    # Additional column transformations
    df['dpb1npgp'] = pd.NA
    df.loc[df['permiss3'].isin([0, 1]) & (df['dpb1nonpermissive'] == 0), 'dpb1npgp'] = 0
    df.loc[df['permiss3'].isin([2, 3]) & (df['dpb1nonpermissive'] == 1), 'dpb1npgp'] = 1
    df['dpb1npgp2'] = pd.cut(df['dpb1nonpermissive'], 
                             bins=[0, 0.05, 0.15, 0.85, 0.95, 1], 
                             labels=[0, 0.15, 0.85, 0.95, 1],
                             right=False,
                             include_lowest=True).astype(float)
    df.loc[df['mpn'], 'disease'] = 51
    df.loc[df['disease'] != 20, 'cytogeneall'] = 'E'
    df['randomuni'] = pd.NA  # Placeholder, need to implement random number generator
    
    # More transformations based on the given SAS code
    df['training'] = df['randomuni'] < 0.85
    df['male'] = df['sex'] == 1

    # Adjust race groupings based on provided conditions
    df.loc[df['racegp'] == 4, 'racegp'] = 3
    df.loc[df['urdbmpbdrace'] == 4, 'urdbmpbdrace'] = 3

    # Adjust KPS and HCTCI based on conditions
    df['kps'] = df['karnofraw']
    df.loc[df['karnofraw'].isna(), 'kps'] = 90
    df['hctci'] = df['coorgscore']
    df.loc[df['coorgscore'].isna(), 'hctci'] = 2

    # Adjust income and intdxtx based on conditions
    df['income'] = df['median_income']
    df.loc[df['median_income'].isna(), 'income'] = 60162
    df['intdxtx'] = df['indxtx']
    df.loc[df['indxtx'].isna(), 'intdxtx'] = 7.14

    # Adjust rcmvpr
    df.loc[df['rcmvpr'] == 7, 'rcmvpr'] = 99
    
    # Adjust disease values
    df.loc[df['disease'] >= 500, 'disease'] = 900
    
    # Adjust aldist values
    df.loc[df['aldist'] == '.E', 'aldist'] = 0
    
    # Adjust cytogeneelnt based on disease value
    df.loc[(df['cytogeneelnt'] == 99) & (df['disease'] != 10), 'cytogeneelnt'] = 0
    
    # Adjust bcrrespr
    df.loc[df['bcrrespr'] == 3, 'bcrrespr'] = 99
    
    # Adjust cmldist
    df.loc[df['cmldist'] == 4, 'cmldist'] = 3
    
    # Adjust clldist
    df['clldist'] = df['clldist'].replace({6: 2, 4: 3})
    
    # Adjust lymdist to create new column lymresist
    df['lymresist'] = df['lymdist'].apply(lambda x: 1 if x == 3 else (0 if x != '.E' else '.E'))
    
    # Adjust lymsubgp
    df.loc[df['lymsubgp'] == 6, 'lymsubgp'] = 99
    
    # Adjust myedist
    df['myedist'] = df['myedist'].replace({1: 2, 5: 4})
    
    # ... [other transformations]
    
    # Adjust gvhprhrx values
    df['gvhprhrx'] = df['gvhprhrx'].replace({
        0: 60,
        1: 2,
        3: 4,
        5: 6,
        13: 14,
        9: 10
    })
    
    # Adjust tbigp values
    df.loc[df['tbigp'].isin([1, 2, 3]), 'tbigp'] = 1
    
    # New column: dqb1match
    df['dqb1match'] = df['himatchdqb1_1'].apply(lambda x: 1 if x == 2 else (0 if not pd.isna(x) else np.nan))
    
    # Adjusting fult12 and efsfult12 based on conditions
    df['fult12'] = np.where((df['dead'] == 0) & (df['intxsurv'] < 12), 1, 0)
    df['efsfult12'] = np.where((df['efs'] == 0) & (df['intxefs'] < 12), 1, 0)
    
    return df

# Call the transformation function
transformed_df = transform_data(df)
