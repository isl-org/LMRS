from random_search.mujoco_random_search_learned import MujocoRandomSearchLearned
import click
import json
import gym

@click.command()
@click.option('--param_file', default='params.json', help='JSON file for exp parameters')
def train_dfo(param_file):
    with open(param_file) as json_params:
        params = json.load(json_params)

    exp_identifier = '|'.join('{}={}'.format(key,val) for (key,val) in params.items())
    params['exp_id'] = exp_identifier

    env = gym.make(params['env_name'])
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # set policy parameters. Possible filters: 'MeanStdFilter' for v2, 'NoFilter' for v1.
    policy_params={'type':'linear',
                   'ob_filter': "MeanStdFilter",
                   'ob_dim':obs_dim,
                   'ac_dim':act_dim}
    params["policy_params"] = policy_params
    params["dimension"] = obs_dim*act_dim

    model = MujocoRandomSearchLearned(params)
    model.search(params['n_iter'], params['every_val'])


if __name__ == '__main__':
    train_dfo()

