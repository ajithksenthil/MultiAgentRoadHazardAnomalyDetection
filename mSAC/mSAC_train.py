for episode in range(max_episodes):
    state = env.reset()
    for t in range(max_timesteps):
        actions = [agent.select_action(state[i]) for i, agent in enumerate(agents)]
        next_state, rewards, done, _ = env.step(actions)

        # Store transitions in replay buffer
        # ...

        # Update parameters of each agent
        for agent in agents:
            batch = replay_buffer.sample(batch_size)
            agent.update_parameters(batch)

        if done:
            break
