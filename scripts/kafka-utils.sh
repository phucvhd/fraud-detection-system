#!/bin/bash

# Kafka utility functions
KAFKA_SERVER="localhost:9092"

# Function to describe a topic
describe_topic() {
    kafka-topics --describe --topic $1 --bootstrap-server $KAFKA_SERVER
}

# Function to consume messages from beginning
consume_from_beginning() {
    kafka-console-consumer --topic $1 \
      --from-beginning --bootstrap-server $KAFKA_SERVER
}

# Function to produce test messages
produce_test_message() {
    echo "Enter messages (Ctrl+C to exit):"
    kafka-console-producer --topic $1 --bootstrap-server $KAFKA_SERVER
}

# Function to delete a topic
delete_topic() {
    kafka-topics --delete --topic $1 --bootstrap-server $KAFKA_SERVER
}

# Function to show consumer groups
show_consumer_groups() {
    kafka-consumer-groups --bootstrap-server $KAFKA_SERVER --list
}

# Function to reset consumer group offset
reset_consumer_group() {
    kafka-consumer-groups --bootstrap-server $KAFKA_SERVER \
      --group $1 --reset-offsets --to-earliest --topic $2 --execute
}

# Usage examples
echo "Available functions:"
echo "  describe_topic <topic-name>"
echo "  consume_from_beginning <topic-name>"
echo "  produce_test_message <topic-name>"
echo "  delete_topic <topic-name>"
echo "  show_consumer_groups"
echo "  reset_consumer_group <group-name> <topic-name>"