#!/bin/bash

KAFKA_CONTAINER="kafka"
KAFKA_BOOTSTRAP_SERVER="localhost:9092"
CREATED_TOPICS=()
FAILED_TOPICS=()

# Function to run kafka commands inside container
kafka_cmd() {
    docker exec $KAFKA_CONTAINER kafka-topics "$@"
}

# Function to create topic and verify
create_and_verify_topic() {
    local topic_name=$1
    local partitions=$2
    local replication_factor=$3
    local config=$4

    echo "Creating topic: $topic_name..."

    # Create the topic
    if [ -n "$config" ]; then
        kafka_cmd --create --if-not-exists --topic $topic_name \
            --bootstrap-server $KAFKA_BOOTSTRAP_SERVER \
            --partitions $partitions --replication-factor $replication_factor \
            --config $config > /dev/null 2>&1
    else
        kafka_cmd --create --if-not-exists --topic $topic_name \
            --bootstrap-server $KAFKA_BOOTSTRAP_SERVER \
            --partitions $partitions --replication-factor $replication_factor > /dev/null 2>&1
    fi

    # Wait a moment for topic to be fully created
    sleep 1

    # Verify topic exists
    if kafka_cmd --list --bootstrap-server $KAFKA_BOOTSTRAP_SERVER | grep -q "^$topic_name$"; then
        echo "  ✅ $topic_name created successfully"
        CREATED_TOPICS+=($topic_name)

        # Verify topic configuration
        echo "  📋 Verifying configuration..."
        kafka_cmd --describe --topic $topic_name --bootstrap-server $KAFKA_BOOTSTRAP_SERVER | grep -E "(Topic:|PartitionCount:|ReplicationFactor:)" | sed 's/^/    /'

    else
        echo "  ❌ Failed to create $topic_name"
        FAILED_TOPICS+=($topic_name)
    fi
    echo
}

# Check if Kafka container is running
echo "🔍 Checking Kafka container status..."
if ! docker ps | grep -q $KAFKA_CONTAINER; then
    echo "❌ Kafka container is not running"
    echo "   Start with: docker-compose up -d"
    exit 1
fi

# Check if Kafka is accessible
echo "🔍 Checking Kafka connectivity..."
if ! kafka_cmd --bootstrap-server $KAFKA_BOOTSTRAP_SERVER --list > /dev/null 2>&1; then
    echo "❌ Cannot connect to Kafka at $KAFKA_BOOTSTRAP_SERVER"
    echo "   Make sure Kafka is fully started (may take 30-60 seconds)"
    exit 1
fi
echo "✅ Kafka is accessible"
echo

echo "🚀 Creating Kafka topics for fraud detection pipeline..."
echo "=================================================="
echo

# Create all topics
create_and_verify_topic "raw-transactions" 12 1 "retention.ms=259200000"
create_and_verify_topic "enriched-transactions" 12 1
create_and_verify_topic "model-predictions" 12 1
create_and_verify_topic "fraud-alerts" 6 1
create_and_verify_topic "transaction-decisions" 6 1

# Summary
echo "📊 SUMMARY"
echo "=========="

if [ ${#CREATED_TOPICS[@]} -gt 0 ]; then
    echo "✅ Successfully created topics (${#CREATED_TOPICS[@]}):"
    for topic in "${CREATED_TOPICS[@]}"; do
        echo "   - $topic"
    done
fi

if [ ${#FAILED_TOPICS[@]} -gt 0 ]; then
    echo "❌ Failed to create topics (${#FAILED_TOPICS[@]}):"
    for topic in "${FAILED_TOPICS[@]}"; do
        echo "   - $topic"
    done
fi

echo
echo "📋 All available topics:"
kafka_cmd --list --bootstrap-server $KAFKA_BOOTSTRAP_SERVER | sed 's/^/   - /'

# Final status
if [ ${#FAILED_TOPICS[@]} -eq 0 ]; then
    echo
    echo "🎉 All topics created successfully! Fraud detection pipeline is ready."
    exit 0
else
    echo
    echo "⚠️  Some topics failed to create. Please check the errors above."
    exit 1
fi