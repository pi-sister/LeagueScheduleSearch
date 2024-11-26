import pandas as pd

# df_gslots = pd.read_csv('test_game_slots.csv', index_col=0)
# df_pslots = pd.read_csv('test_practice_slots.csv', index_col=0)
df_events = pd.read_csv('test_events.csv', index_col=0)

print(df_events[df_events.Type == 'P'].index[0])

df_practices = df_events[df_events.Type == 'P'].index

print(len(df_practices))

labels = ['CMSAU13T3DIV02','CMSAU17T1DIV01','CUSAO18DIV01PRC01']

row_numbers = [df_events.index.get_loc(label) for label in labels]

row_numbers.sort(reverse=True)

print(row_numbers)

str_1 = 'CMSAU13T3DIV01PRC01'
str_2 =  'PRC'

print(str_1.partition(str_2)[0])

print(df_events.index[2])
