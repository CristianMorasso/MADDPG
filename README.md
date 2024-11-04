# MADDPG implementation that allows communication among agents

### Work in progress
<hr>

 
Param `--comm_type` identifies the communication protocol can be `[None, broadcast, vel_comm, directEsc, closer_target]` with `None` no communication is involved.

See *argParser.py* to all parameters.

The code work but need to be cleaned up.\
`par_sharing` with dial not available, pls do not use that parameter.
`LSTM` not tested yet.