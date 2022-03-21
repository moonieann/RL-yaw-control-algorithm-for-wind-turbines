#!/usr/bin/python
# -*- coding: utf-8 -*-
from yaw_RL_module import *


# --------------------------------------------------------------------------

    

power_curve = pd.read_excel('power_curve.xlsx')
dataset_file = 'steady_wind.csv'
(wind_timeseries, wind_timeseries_not_agg) = get_dataset_from_simu(dataset_file,
                                                                   cycle_period=10,
                                                                   rolling_average_duration=20)
    
    




yaw_params = {
    'yaw_rate_max': 0.3,
    'yaw_consumption':18,
    'rated_speed': 14,
    'ref_speed': power_curve['ref_v'].to_list(),
    'ref_power': power_curve['ref_P'].to_list(),
    'cycle_period': 10,
    'w2': 40,
    }

model_params = {
    'wind_timeseries': wind_timeseries,
    'wind_timeseries_not_agg': wind_timeseries_not_agg,
    'start_index': 100,
    'stop_index': 1100,    
    'start_index_test': 1100,
    'stop_index_test': 2100,
    'filter_duration': 1,
    'yaw_params': yaw_params,
    'ancestors': 12,
    'training_steps': 500000,
    }

experiment = Experiment(api_key='GYoAMnAcbnbZ9p1PurkZCaSX0',
                        project_name='yaw-rl',
                        workspace='albanpuech')


experiment.log_parameters(model_params)
experiment.log_code(os.path.basename(__file__))
experiment.log_code('yaw_RL_module.py')
name = 'steady'
experiment.set_name(name)

env = YawEnv(
    wind_timeseries,
    model_params['start_index'],
    model_params['stop_index'],
    model_params['ancestors'],
    model_params['filter_duration'],
    yaw_params,
    )
 
model = PPO('MlpPolicy', env, verbose=1)
logger_callback = Cometlogger(experiment, model_params,
                              eval_freq=20000)
callback = CallbackList([logger_callback])
model.learn(total_timesteps=model_params['training_steps'],callback=callback)
#model_eval = PPO.load('steady_wind')


(res_model, nac_pos_model, power_improvement, power_control, power_simu) = test_model(
    model,
    model_params['wind_timeseries'],
    model_params['start_index_test'],
    model_params['stop_index_test'],
    model_params['ancestors'],
    model_params['filter_duration'],
    model_params['yaw_params'],
    experiment=experiment,
    datatype='test',
    )

(res_baseline_simu, nac_pos_baseline_simu, wind_dir) = test_trad_control(
    model_params['wind_timeseries'],
    model_params['wind_timeseries_not_agg'],
    yaw_params['cycle_period'],
    model_params['start_index_test'],
    model_params['stop_index_test'],
    experiment=experiment,
    datatype='baseline_simu',
    )

(res_baseline_logs, nac_pos_baseline_logs, wind_dir) = test_trad_control(
    model_params['wind_timeseries'],
    model_params['wind_timeseries_not_agg'],
    yaw_params['cycle_period'],
    model_params['start_index_test'],
    model_params['stop_index_test'],
    experiment=experiment,
    datatype='baseline_logs',
    )


experiment.end()


power_prod_change, conso_yaw_change, net_prod_change, rel_net_prod_change,yaw_error_rel_change = get_comparaison_metrics(wind_dir,power_control,power_simu,nac_pos_model, nac_pos_baseline_simu, yaw_params['yaw_rate_max'], yaw_params['yaw_consumption'], 50)    
fig, axs = plt.subplots(6, sharex=True, figsize=(15,25), gridspec_kw={'height_ratios': [3, 1, 1,1,1,1]})
plt.xlabel('time (sec)', fontsize=20)
plt.xlim(0,(model_params['stop_index_test']- model_params['start_index_test'])*yaw_params['cycle_period'])

axs[0].plot(range(0,(model_params['stop_index_test']- model_params['start_index_test'])*yaw_params['cycle_period'] ,yaw_params['cycle_period']),wind_dir,label='wind direction')
axs[0].plot(range(0,(model_params['stop_index_test']- model_params['start_index_test'])*yaw_params['cycle_period'] ,yaw_params['cycle_period']),nac_pos_model,linewidth=4.0, label='nacelle position RL algorithm')
axs[0].plot(range(0,(model_params['stop_index_test']- model_params['start_index_test'])*yaw_params['cycle_period'],yaw_params['cycle_period'] ),nac_pos_baseline_simu,linewidth=4.0, label='nacelle position simulated baseline algorithm')
axs[0].plot(range(0,(model_params['stop_index_test']- model_params['start_index_test'])*yaw_params['cycle_period'] ,yaw_params['cycle_period']),nac_pos_baseline_logs,linewidth=4.0, label='nacelle position baseline algorithm from logs')
axs[0].legend(fontsize=20)
axs[0].tick_params(labelsize=20)
axs[0].set_ylabel(ylabel='angle (deg)', fontsize=20)

axs[1].bar(range(0,(model_params['stop_index_test']- model_params['start_index_test'])*yaw_params['cycle_period'] ,yaw_params['cycle_period']*50),yaw_error_rel_change,500,linewidth=4.0, label='RL alg', align='edge')
axs[1].get_yaxis().set_major_formatter(FuncFormatter(lambda y, _: "{:.1f}%".format(y)))
axs[1].legend(fontsize=20)
axs[1].tick_params(labelsize=20)
axs[1].set_ylabel(ylabel='yaw misaligment \ndecrease (per cent)', fontsize=20,)

axs[2].bar(range(0,(model_params['stop_index_test']- model_params['start_index_test'])*yaw_params['cycle_period'] ,yaw_params['cycle_period']*50),power_prod_change,500,linewidth=4.0, label='RL alg', align='edge')
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: "{:.1f}%".format(y)))
axs[2].legend(fontsize=20)
axs[2].tick_params(labelsize=20)
axs[2].set_ylabel(ylabel='power output \nincrease (kW)', fontsize=20,)

axs[3].bar(range(0,(model_params['stop_index_test']- model_params['start_index_test'])*yaw_params['cycle_period'] ,yaw_params['cycle_period']*50),conso_yaw_change,500,linewidth=4.0, label='RL alg', align='edge')
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: "{:.1f}%".format(y)))
axs[3].legend(fontsize=20)
axs[3].tick_params(labelsize=20)
axs[3].set_ylabel(ylabel='yaw consumption \nincrease (kW)', fontsize=20,)

axs[4].bar(range(0,(model_params['stop_index_test']- model_params['start_index_test'])*yaw_params['cycle_period'] ,yaw_params['cycle_period']*50),net_prod_change,500,linewidth=4.0, label='RL alg', align='edge')
axs[4].legend(fontsize=20)
axs[4].tick_params(labelsize=20)
axs[4].set_ylabel(ylabel='net power output \nincrease (kW)', fontsize=20,)

axs[5].bar(range(0,(model_params['stop_index_test']- model_params['start_index_test'])*yaw_params['cycle_period'] ,yaw_params['cycle_period']*50),rel_net_prod_change,500,linewidth=4.0, label='RL alg', align='edge')
axs[5].get_yaxis().set_major_formatter(FuncFormatter(lambda y, _: "{:.1f}%".format(y)))
axs[5].legend(fontsize=20)
axs[5].tick_params(labelsize=20)
axs[5].set_ylabel(ylabel='net power output \nincrease (per cent)', fontsize=20,)

fig.tight_layout()
plt.savefig('steady_dataset',dpi=600)

    



fig, axs = plt.subplots(2, sharex=True)
axs[0].plot(range(model_params['start_index']*yaw_params['cycle_period'],model_params['stop_index']*yaw_params['cycle_period'], yaw_params['cycle_period'] ),wind_timeseries['wind_direction'][model_params['start_index']:model_params['stop_index']],label='train')
axs[0].plot(range(model_params['start_index_test']*yaw_params['cycle_period'],model_params['stop_index_test']*yaw_params['cycle_period'],yaw_params['cycle_period'] ),wind_timeseries['wind_direction'][model_params['start_index_test']: model_params['stop_index_test']],label='test')
plt.xlabel('time (sec)')
axs[1].plot(range(model_params['start_index']*yaw_params['cycle_period'],model_params['stop_index']*yaw_params['cycle_period'], yaw_params['cycle_period'] ),wind_timeseries['wind_speed'][model_params['start_index']:model_params['stop_index']],label='train')
axs[1].plot(range(model_params['start_index_test']*yaw_params['cycle_period'],model_params['stop_index_test']*yaw_params['cycle_period'],yaw_params['cycle_period'] ),wind_timeseries['wind_speed'][model_params['start_index_test']: model_params['stop_index_test']],label='test')
plt.setp(axs[1], ylabel='wind speed (m/s)')
plt.setp(axs[0], ylabel='wind direction (deg)')
plt.legend()
plt.savefig('steady_results',dpi=1000)
    
print(res_baseline_simu)
print(res_baseline_logs)
print(res_model)
    









