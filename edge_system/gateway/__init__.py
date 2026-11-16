"""
Decision-support gateway.

The one service the operator dashboard talks to. It owns the three things that
are new to the deployed system - price rules, price recommendations, and the
operator's decisions on them - and reads everything else (forecasts, stock,
model health) from the edge nodes, the inventory service and the control plane.

It deliberately sits OUTSIDE the experiment path. The nodes still answer /price
with the raw PPO decision, the simulator still sells at whatever it sells at,
and nothing an operator approves or rejects flows back into a run. A human in
the loop would be an uncontrolled variable in E1/E2; here it is a demo feature
layered on top.
"""
