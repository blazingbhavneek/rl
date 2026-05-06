## SCADA API Reference

## Ring Buffer

- RingBufferInsert — Inserts a data element into a circular ring buffer managed on the subordinate processor. The buffer overwrites the oldest entry when full. (Usage: Called with the element value to enqueue.; Interacts with: mpf_mfs_enq, subordinate processor buffer state)

- RingBufferRead — Reads and removes the oldest element from the subordinate processor ring buffer. Returns the element value. (Usage: Called with no arguments to dequeue the head element.; Interacts with: mpf_mfs_deq, subordinate processor buffer state)

## State Registers

- query_secondary_transfer_progress — Simulates verification of asynchronous multi-write operation status on a subordinate processor. (Usage: Invoked with no arguments to query the current execution state of background write tasks.; Interacts with: Subordinate processor state, Main processor state)

- latch_register_value — Captures and holds the current value of a sensor register so that subsequent reads return the latched snapshot rather than live data. (Usage: Called with no arguments to freeze the register.; Interacts with: Sensor register, latch enable line)

## Event / Notification

- notify_main_processor — Signals the main processor that a subordinate event has occurred and data is ready for collection. (Usage: Called by subordinate after completing a write sequence.; Interacts with: interrupt line, main processor event queue)

- wait_for_subordinate_ack — Blocks the calling process until the subordinate processor acknowledges receipt of a command. (Usage: Called after issuing a write command.; Interacts with: subordinate ACK line, timeout counter)
