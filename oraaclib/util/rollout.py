"""Helper functions to conduct a rollout with policies or agents."""


def oraac_rollout(env, agent, gradient_steps, max_episodes,
                  max_episode_steps=1000,
                  eval_freq=100,
                  times_eval=5):
    for grad_step in range(int(gradient_steps)):
        agent.train()

        if grad_step % eval_freq == 0:
            agent.evaluate_model(max_episode_steps,
                                 times_eval=times_eval)
            agent.save_model()

        if agent.eval_episode >= max_episodes:
            break
    agent.save_final_model()
    agent.end_interaction()
    env.close()
