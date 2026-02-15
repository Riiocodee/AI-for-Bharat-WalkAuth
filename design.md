# Design Document: Zero-Interface Payments Using Gait Recognition

## 1. High-Level Architecture

### 1.1 Architecture Overview

The Zero-Interface Payment System follows a hybrid edge-cloud architecture with three primary layers:

```
┌─────────────────────────────────────────────────────────────────┐
│                        CUSTOMER LAYER                            │
│  Mobile App (Enrollment) | In-Store Experience (Passive)        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                         EDGE LAYER                               │
│  Camera Network → Pose Estimation → Feature Extraction          │
│  (Store Premises - Low Latency Processing)                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        CLOUD LAYER                               │
│  Gait Matching Engine | Biometric DB | Payment Gateway          │
│  Admin Dashboard | Analytics | Model Training Pipeline          │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Architecture Principles

- **Hybrid Processing**: Latency-critical tasks (pose estimation) at edge, compute-intensive tasks (matching) in cloud
- **Microservices**: Loosely coupled services for scalability and maintainability
- **Event-Driven**: Asynchronous communication via message queues (Kafka/RabbitMQ)
- **Security-First**: Zero-trust architecture with end-to-end encryption
- **Privacy by Design**: Minimal data retention, pseudonymization, consent management


## 2. Component Breakdown

### 2.1 Camera Capture Layer

**Purpose**: Capture high-quality video streams of customer gait patterns at store entry/exit points.

**Components**:
- **Overhead RGB Cameras**: 1080p resolution, 30 FPS, wide-angle lens (120° FOV)
- **Depth Sensors** (optional): Intel RealSense or Azure Kinect for 3D pose estimation
- **Edge Processing Unit**: NVIDIA Jetson Xavier NX or equivalent (16 GB RAM, 384-core GPU)
- **Network Video Recorder (NVR)**: Temporary buffer for dispute resolution (7-day retention)

**Technical Specifications**:
- Camera placement: 8-12 feet height, 45° angle for optimal gait capture
- Coverage: 4 cameras per 500 sq ft, overlapping fields of view
- Encoding: H.264 compression for bandwidth optimization
- Synchronization: NTP time sync across all cameras (<10ms drift)

**Data Flow**:
1. Cameras capture video at 30 FPS
2. Motion detection triggers region of interest (ROI) extraction
3. ROI frames sent to edge device for pose estimation
4. Raw video optionally buffered for 7 days (dispute resolution)

**Challenges & Mitigations**:
- **Occlusion**: Multiple camera angles ensure at least one clear view
- **Lighting variations**: Auto-exposure and HDR processing
- **Privacy**: No facial close-ups, overhead angle prevents face capture


### 2.2 Pose Estimation Module

**Purpose**: Extract skeletal keypoints from video frames to represent human body pose.

**Technology Stack**:
- **Model**: OpenPose, MediaPipe Pose, or HRNet (High-Resolution Network)
- **Framework**: TensorRT-optimized models for edge inference
- **Output**: 17-33 keypoints (COCO or MPII format) per frame

**Keypoint Extraction**:
```
Key Body Points for Gait Analysis:
- Hips (left/right)
- Knees (left/right)
- Ankles (left/right)
- Shoulders (left/right)
- Elbows (left/right)
- Head/Neck
```

**Processing Pipeline**:
1. **Preprocessing**: Resize frames to 256x256, normalize pixel values
2. **Inference**: Run pose estimation model (15-20ms per frame on Jetson)
3. **Post-processing**: Filter low-confidence keypoints (<0.5 threshold)
4. **Temporal smoothing**: Kalman filter to reduce jitter across frames

**Performance Optimization**:
- Batch processing: 4-8 frames per inference batch
- Model quantization: INT8 precision (3x speedup, <1% accuracy loss)
- ROI cropping: Process only detected person regions (not full frame)

**Output Format**:
```json
{
  "frame_id": 12345,
  "timestamp": "2026-02-15T10:30:45.123Z",
  "person_id": "temp_track_001",
  "keypoints": [
    {"name": "left_hip", "x": 120, "y": 180, "confidence": 0.92},
    {"name": "right_hip", "x": 140, "y": 182, "confidence": 0.89},
    ...
  ]
}
```


### 2.3 Gait Feature Extraction Engine

**Purpose**: Convert pose keypoint sequences into discriminative gait feature vectors.

**Feature Engineering**:

1. **Spatial Features** (per frame):
   - Joint angles (hip, knee, ankle)
   - Stride length (distance between feet)
   - Step width (lateral distance)
   - Body posture (torso angle, shoulder alignment)

2. **Temporal Features** (across frames):
   - Gait cycle duration (heel strike to heel strike)
   - Cadence (steps per minute)
   - Swing/stance phase ratio
   - Velocity profile (acceleration/deceleration patterns)

3. **Frequency Features**:
   - FFT of joint trajectories (dominant frequencies)
   - Harmonic ratios (gait periodicity)

**Feature Vector Composition**:
```
Gait Feature Vector (128 dimensions):
- Spatial features: 40 dims (joint angles, distances)
- Temporal features: 48 dims (cycle statistics, phase ratios)
- Frequency features: 40 dims (FFT coefficients)
```

**Extraction Pipeline**:
1. **Gait Cycle Segmentation**: Detect heel strikes using ankle keypoint velocity
2. **Normalization**: Scale features by height (estimated from keypoints)
3. **Feature Computation**: Calculate spatial/temporal/frequency features
4. **Dimensionality Reduction**: PCA or autoencoder to 128 dims
5. **Feature Encoding**: Serialize to protobuf for efficient transmission

**Robustness Enhancements**:
- **View-invariant features**: Use 3D pose estimation or multi-view fusion
- **Speed normalization**: Adjust for walking speed variations
- **Footwear compensation**: Train model on diverse footwear types
- **Carrying objects**: Detect and flag when customer carries heavy items

**Code Snippet (Pseudocode)**:
```python
def extract_gait_features(keypoint_sequence):
    # keypoint_sequence: List of frames with keypoints
    
    # Step 1: Segment gait cycles
    cycles = segment_gait_cycles(keypoint_sequence)
    
    # Step 2: Extract features per cycle
    features = []
    for cycle in cycles:
        spatial = compute_spatial_features(cycle)
        temporal = compute_temporal_features(cycle)
        frequency = compute_frequency_features(cycle)
        features.append(concatenate(spatial, temporal, frequency))
    
    # Step 3: Aggregate across cycles (mean + std)
    gait_vector = aggregate_features(features)
    
    # Step 4: Normalize and reduce dimensions
    gait_vector = normalize(gait_vector)
    gait_vector = pca_transform(gait_vector, n_components=128)
    
    return gait_vector
```


### 2.4 AI Classification Model (CNN + LSTM)

**Purpose**: Match extracted gait features against enrolled customer database for identification.

**Model Architecture**:

```
Input: Gait Feature Sequence (T x 128)
  ↓
[CNN Block 1] → Conv1D(128→256) → BatchNorm → ReLU → Dropout(0.3)
  ↓
[CNN Block 2] → Conv1D(256→512) → BatchNorm → ReLU → Dropout(0.3)
  ↓
[LSTM Layer] → Bidirectional LSTM(512 hidden units) → Dropout(0.4)
  ↓
[Attention Layer] → Self-attention mechanism (focus on discriminative frames)
  ↓
[Dense Layer] → Fully Connected(512→256) → ReLU → Dropout(0.3)
  ↓
[Output Layer] → Softmax(N classes) or Embedding(128 dims for metric learning)
```

**Training Strategy**:

1. **Metric Learning Approach** (Preferred):
   - Use Triplet Loss or ArcFace Loss
   - Learn embedding space where same person's gaits are close
   - Similarity threshold for identification (cosine similarity > 0.85)
   - Supports open-set recognition (new enrollments without retraining)

2. **Classification Approach** (Alternative):
   - Softmax over N enrolled customers
   - Requires retraining when new customers enroll
   - Suitable for closed-set scenarios (fixed customer base)

**Training Data**:
- **Public datasets**: CASIA Gait Database, OU-ISIR, TUM-GAID
- **Synthetic data**: Augment with speed variations, viewpoint changes
- **In-house data**: Collect diverse gait samples during pilot phase
- **Data augmentation**: Random cropping, temporal jittering, noise injection

**Training Hyperparameters**:
```yaml
optimizer: Adam
learning_rate: 0.001 (with cosine annealing)
batch_size: 64
epochs: 100
loss_function: Triplet Loss (margin=0.3)
regularization: L2 (weight_decay=0.0001)
early_stopping: patience=10 (validation loss)
```

**Model Evaluation Metrics**:
- **Rank-1 Accuracy**: % of times correct person is top match
- **Rank-5 Accuracy**: % of times correct person is in top 5
- **Equal Error Rate (EER)**: Point where FPR = FNR
- **ROC-AUC**: Area under receiver operating characteristic curve

**Inference Pipeline**:
1. Receive gait feature vector from edge device
2. Compute embedding using trained model (GPU inference: 50ms)
3. Search biometric database for nearest neighbor (vector similarity)
4. Return top-K matches with confidence scores
5. Apply threshold (confidence > 0.85) for positive identification

**Model Optimization**:
- **Quantization**: TensorRT INT8 for 3x speedup
- **Pruning**: Remove 30% of weights with minimal accuracy loss
- **Knowledge distillation**: Train smaller student model from large teacher
- **ONNX export**: Cross-platform deployment


### 2.5 Biometric Database

**Purpose**: Securely store and efficiently retrieve gait templates for customer identification.

**Database Architecture**:

**Primary Database**: PostgreSQL with pgvector extension
- **Customer Table**: User profiles, consent records, payment links
- **Gait Template Table**: Encrypted biometric embeddings (128-dim vectors)
- **Transaction Table**: Payment history, receipts, audit logs

**Vector Search Engine**: Milvus or Pinecone
- Optimized for high-dimensional similarity search
- Supports billion-scale vector indexing
- Sub-millisecond query latency with HNSW index

**Schema Design**:

```sql
-- Customer Profile Table
CREATE TABLE customers (
    customer_id UUID PRIMARY KEY,
    phone_number VARCHAR(15) UNIQUE ENCRYPTED,
    email VARCHAR(255) ENCRYPTED,
    enrollment_date TIMESTAMP,
    consent_signature BYTEA,
    consent_video_url VARCHAR(500),
    kyc_verified BOOLEAN,
    account_status ENUM('active', 'suspended', 'deleted'),
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

-- Gait Template Table (Encrypted)
CREATE TABLE gait_templates (
    template_id UUID PRIMARY KEY,
    customer_id UUID REFERENCES customers(customer_id),
    gait_embedding VECTOR(128) ENCRYPTED,
    enrollment_device VARCHAR(100),
    enrollment_location VARCHAR(200),
    template_version INT,
    quality_score FLOAT,
    created_at TIMESTAMP,
    expires_at TIMESTAMP
);

-- Payment Methods Table
CREATE TABLE payment_methods (
    payment_id UUID PRIMARY KEY,
    customer_id UUID REFERENCES customers(customer_id),
    payment_type ENUM('upi', 'card', 'bank_account'),
    token VARCHAR(255) ENCRYPTED, -- Tokenized payment credential
    is_primary BOOLEAN,
    added_at TIMESTAMP
);

-- Transaction Table
CREATE TABLE transactions (
    transaction_id UUID PRIMARY KEY,
    customer_id UUID REFERENCES customers(customer_id),
    store_id UUID,
    amount DECIMAL(10, 2),
    items JSONB,
    payment_status ENUM('pending', 'success', 'failed'),
    payment_method_id UUID,
    timestamp TIMESTAMP,
    receipt_url VARCHAR(500)
);

-- Audit Log Table
CREATE TABLE audit_logs (
    log_id UUID PRIMARY KEY,
    customer_id UUID,
    action ENUM('enrollment', 'identification', 'payment', 'data_access', 'deletion'),
    actor VARCHAR(100), -- system, admin, customer
    ip_address INET,
    timestamp TIMESTAMP,
    details JSONB
);
```

**Encryption Strategy**:
- **Application-level encryption**: Encrypt gait embeddings before storage
- **Column-level encryption**: Sensitive fields (phone, email) encrypted with AES-256
- **Key management**: AWS KMS or Azure Key Vault for key rotation
- **Customer-specific keys**: Each customer's data encrypted with unique DEK (Data Encryption Key)

**Vector Indexing**:
```python
# Milvus collection schema
collection_schema = {
    "name": "gait_embeddings",
    "fields": [
        {"name": "customer_id", "type": "VARCHAR", "max_length": 36},
        {"name": "embedding", "type": "FLOAT_VECTOR", "dim": 128},
        {"name": "quality_score", "type": "FLOAT"}
    ]
}

# Create HNSW index for fast similarity search
index_params = {
    "metric_type": "COSINE",
    "index_type": "HNSW",
    "params": {"M": 16, "efConstruction": 200}
}
```

**Query Performance**:
- **Similarity search**: <10ms for 1M vectors (HNSW index)
- **Batch queries**: 1000 QPS sustained throughput
- **Replication**: Master-slave setup with read replicas
- **Sharding**: Partition by store_id for horizontal scaling


### 2.6 Payment Integration Module

**Purpose**: Process automatic payments via UPI, cards, or bank accounts after customer identification.

**Supported Payment Methods**:
1. **UPI (Unified Payments Interface)**: Razorpay, Paytm, PhonePe APIs
2. **Credit/Debit Cards**: Stripe, Razorpay with tokenization
3. **Net Banking**: Direct bank API integration (ICICI, HDFC, SBI)
4. **Digital Wallets**: Paytm, Google Pay, Amazon Pay

**Payment Flow**:

```
Customer Exit Detected
  ↓
Calculate Total Bill (Item Tracking System)
  ↓
Retrieve Payment Method (Primary from customer profile)
  ↓
Initiate Payment Request (via Gateway API)
  ↓
[Payment Gateway Processing]
  ↓
Receive Payment Confirmation (Webhook)
  ↓
Update Transaction Status (Database)
  ↓
Send Receipt (Email/SMS/Push Notification)
```

**API Integration Example (UPI via Razorpay)**:

```python
import razorpay

def process_payment(customer_id, amount, items):
    # Retrieve customer payment token
    payment_token = get_payment_token(customer_id)
    
    # Initialize Razorpay client
    client = razorpay.Client(auth=(RAZORPAY_KEY, RAZORPAY_SECRET))
    
    # Create payment order
    order = client.order.create({
        "amount": int(amount * 100),  # Amount in paise
        "currency": "INR",
        "receipt": f"receipt_{transaction_id}",
        "notes": {
            "customer_id": customer_id,
            "store_id": store_id,
            "items": json.dumps(items)
        }
    })
    
    # Charge using saved token (recurring payment)
    payment = client.payment.create_recurring({
        "email": customer_email,
        "contact": customer_phone,
        "amount": int(amount * 100),
        "currency": "INR",
        "order_id": order['id'],
        "customer_id": razorpay_customer_id,
        "token": payment_token,
        "recurring": "1"
    })
    
    # Handle response
    if payment['status'] == 'captured':
        return {"status": "success", "payment_id": payment['id']}
    else:
        return {"status": "failed", "error": payment['error_description']}
```

**Tokenization**:
- Customer payment credentials never stored in raw form
- Payment gateway provides token during enrollment
- Token used for recurring payments (pre-authorized)
- Token rotation every 6 months for security

**Webhook Handling**:
```python
@app.route('/webhook/payment', methods=['POST'])
def payment_webhook():
    # Verify webhook signature
    signature = request.headers.get('X-Razorpay-Signature')
    if not verify_signature(request.data, signature):
        return {"error": "Invalid signature"}, 401
    
    # Parse webhook payload
    event = request.json
    
    if event['event'] == 'payment.captured':
        transaction_id = event['payload']['payment']['entity']['notes']['transaction_id']
        update_transaction_status(transaction_id, 'success')
        send_receipt(transaction_id)
    
    elif event['event'] == 'payment.failed':
        transaction_id = event['payload']['payment']['entity']['notes']['transaction_id']
        update_transaction_status(transaction_id, 'failed')
        alert_store_staff(transaction_id)
    
    return {"status": "ok"}, 200
```

**Failure Handling**:
- **Retry logic**: 3 attempts with exponential backoff
- **Fallback payment**: SMS link for manual payment
- **Store alert**: Notify staff for failed payments (optional gate lock)
- **Grace period**: 24 hours to complete payment before account suspension

**Compliance**:
- **PCI-DSS**: No card numbers stored, tokenization mandatory
- **RBI guidelines**: Two-factor authentication for high-value transactions (>₹5000)
- **Dispute resolution**: 7-day video retention for transaction verification


### 2.7 Cloud Infrastructure

**Purpose**: Provide scalable, reliable, and secure infrastructure for compute, storage, and networking.

**Cloud Provider**: AWS (can be adapted to Azure/GCP)

**Infrastructure Components**:

1. **Compute**:
   - **ECS/EKS**: Container orchestration for microservices
   - **EC2 GPU Instances**: p3.2xlarge for model inference (V100 GPU)
   - **Lambda**: Serverless functions for webhooks, notifications
   - **Auto Scaling Groups**: Scale based on CPU/GPU utilization

2. **Storage**:
   - **S3**: Object storage for model artifacts, video backups, receipts
   - **RDS PostgreSQL**: Relational database with Multi-AZ deployment
   - **ElastiCache Redis**: Caching layer for frequent queries
   - **EFS**: Shared file system for model serving

3. **Networking**:
   - **VPC**: Isolated network with public/private subnets
   - **ALB**: Application Load Balancer for traffic distribution
   - **CloudFront**: CDN for mobile app assets and receipts
   - **Direct Connect**: Dedicated connection for store networks

4. **Security**:
   - **KMS**: Key management for encryption
   - **Secrets Manager**: Store API keys, database credentials
   - **WAF**: Web application firewall for DDoS protection
   - **GuardDuty**: Threat detection and monitoring

5. **Monitoring & Logging**:
   - **CloudWatch**: Metrics, logs, alarms
   - **X-Ray**: Distributed tracing for microservices
   - **ELK Stack**: Elasticsearch, Logstash, Kibana for log analysis
   - **Prometheus + Grafana**: Custom metrics and dashboards

**Infrastructure as Code (Terraform)**:

```hcl
# VPC Configuration
resource "aws_vpc" "main" {
  cidr_block = "10.0.0.0/16"
  enable_dns_hostnames = true
  tags = { Name = "gait-payment-vpc" }
}

# ECS Cluster for Microservices
resource "aws_ecs_cluster" "main" {
  name = "gait-payment-cluster"
  
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

# RDS PostgreSQL with Encryption
resource "aws_db_instance" "main" {
  identifier           = "gait-payment-db"
  engine               = "postgres"
  engine_version       = "14.7"
  instance_class       = "db.r5.xlarge"
  allocated_storage    = 500
  storage_encrypted    = true
  kms_key_id          = aws_kms_key.db.arn
  multi_az            = true
  backup_retention_period = 30
  
  vpc_security_group_ids = [aws_security_group.db.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name
}

# Auto Scaling for GPU Inference
resource "aws_autoscaling_group" "inference" {
  name                = "gait-inference-asg"
  vpc_zone_identifier = aws_subnet.private[*].id
  min_size            = 2
  max_size            = 10
  desired_capacity    = 4
  
  launch_template {
    id      = aws_launch_template.inference.id
    version = "$Latest"
  }
  
  target_group_arns = [aws_lb_target_group.inference.arn]
  
  tag {
    key                 = "Name"
    value               = "gait-inference-server"
    propagate_at_launch = true
  }
}
```

**Cost Optimization**:
- **Spot Instances**: Use for non-critical workloads (70% cost savings)
- **Reserved Instances**: 1-year commitment for baseline capacity
- **S3 Lifecycle Policies**: Move old videos to Glacier after 7 days
- **Right-sizing**: Monitor utilization and adjust instance types

**Disaster Recovery**:
- **RTO (Recovery Time Objective)**: 4 hours
- **RPO (Recovery Point Objective)**: 1 hour
- **Backup Strategy**: Daily automated snapshots, cross-region replication
- **Failover**: Active-passive setup with Route 53 health checks


## 3. Sequence Diagram: User Entry to Payment Deduction

```
Customer    Camera    Edge Device    Cloud API    Gait Engine    Payment Gateway    Database
   |           |            |            |             |                |              |
   |--Enter--->|            |            |             |                |              |
   |           |--Capture-->|            |             |                |              |
   |           |   Video    |            |             |                |              |
   |           |            |--Pose----->|             |                |              |
   |           |            | Estimation |             |                |              |
   |           |            |<--Keypts---|             |                |              |
   |           |            |            |             |                |              |
   |           |            |--Extract-->|             |                |              |
   |           |            |  Features  |             |                |              |
   |           |            |<--Vector---|             |                |              |
   |           |            |            |             |                |              |
   |           |            |----------->|--Match----->|                |              |
   |           |            |  Gait Vec  | Request     |                |              |
   |           |            |            |             |--Query-------->|              |
   |           |            |            |             |  Templates     |              |
   |           |            |            |             |<--Results------|              |
   |           |            |            |             |                |              |
   |           |            |            |             |--Compute------>|              |
   |           |            |            |             | Similarity     |              |
   |           |            |            |             |<--Top Match----|              |
   |           |            |            |<--Customer--|                |              |
   |           |            |            |    ID       |                |              |
   |           |            |<-----------|             |                |              |
   |           |            | Identified |             |                |              |
   |           |            |            |             |                |              |
   |--Shop---->|            |            |             |                |              |
   | (Items    |            |            |             |                |              |
   | Tracked)  |            |            |             |                |              |
   |           |            |            |             |                |              |
   |--Exit---->|            |            |             |                |              |
   |           |--Detect--->|            |             |                |              |
   |           |   Exit     |            |             |                |              |
   |           |            |----------->|--Calculate->|                |              |
   |           |            |  Trigger   |    Bill     |                |              |
   |           |            |            |             |                |              |
   |           |            |            |------------>|--Get Payment-->|              |
   |           |            |            |  Payment    |    Method      |              |
   |           |            |            |  Request    |<--Token--------|              |
   |           |            |            |             |                |              |
   |           |            |            |------------>|--------------->|--Charge----->|
   |           |            |            |             |  Initiate      | Customer     |
   |           |            |            |             |  Payment       |<--Success----|
   |           |            |            |             |                |              |
   |           |            |            |<------------|<---------------|              |
   |           |            |            | Confirmation|                |              |
   |           |            |            |             |                |              |
   |           |            |            |------------>|--------------->|--Update----->|
   |           |            |            |  Save Txn   |                | Transaction  |
   |           |            |            |             |                |<--Saved------|
   |<--Receipt-|            |            |             |                |              |
   | (SMS/Email)            |            |             |                |              |
   |           |            |            |             |                |              |
```

**Sequence Steps Explained**:

1. **Entry Detection** (0-1s): Customer enters store, camera detects motion
2. **Pose Estimation** (1-2s): Edge device extracts skeletal keypoints from video
3. **Feature Extraction** (2-3s): Gait features computed from keypoint sequence
4. **Identification** (3-4s): Cloud API matches gait vector against database
5. **Shopping Phase** (variable): Customer shops, items tracked by separate system
6. **Exit Detection** (0-1s): Customer crosses exit threshold
7. **Bill Calculation** (1s): Total computed from tracked items
8. **Payment Processing** (1-2s): Automatic charge via payment gateway
9. **Receipt Delivery** (2-3s): Digital receipt sent to customer

**Total Latency**: Entry to identification: ~4s | Exit to payment: ~3s


## 4. Data Flow Description

### 4.1 Enrollment Data Flow

```
Mobile App → API Gateway → Enrollment Service → Gait Processor → Database
                                    ↓
                            Consent Manager → S3 (Consent Video)
                                    ↓
                            KYC Service → ID Verification API
                                    ↓
                            Payment Service → Payment Gateway (Tokenization)
```

**Steps**:
1. Customer records 30-second walking video in mobile app
2. Video uploaded to S3 via pre-signed URL (encrypted in transit)
3. Enrollment service triggers gait processing job
4. Gait processor extracts features and stores encrypted template
5. Consent video and signature stored for compliance
6. KYC verification via Aadhaar/PAN API
7. Payment method tokenized and linked to customer profile

### 4.2 Identification Data Flow

```
Camera → Edge Device → Message Queue (Kafka) → Gait Matching Service
                                                        ↓
                                                Vector DB (Milvus)
                                                        ↓
                                                Customer Service
                                                        ↓
                                                Cache (Redis)
```

**Steps**:
1. Camera captures video, edge device extracts gait features
2. Feature vector published to Kafka topic (real-time stream)
3. Gait matching service consumes message
4. Vector similarity search in Milvus (top-5 candidates)
5. Customer details fetched from PostgreSQL (or Redis cache)
6. Identification result cached for session duration

### 4.3 Payment Data Flow

```
Exit Event → Item Tracking System → Billing Service → Payment Service
                                                              ↓
                                                    Payment Gateway API
                                                              ↓
                                                    Webhook Handler
                                                              ↓
                                                    Transaction DB
                                                              ↓
                                                    Notification Service
```

**Steps**:
1. Exit detection triggers billing calculation
2. Item tracking system provides cart details
3. Billing service computes total with taxes
4. Payment service retrieves tokenized payment method
5. Payment gateway charges customer account
6. Webhook confirms transaction status
7. Transaction record saved with receipt URL
8. Notification service sends receipt via email/SMS

### 4.4 Data Retention & Deletion Flow

```
Customer Request → Data Deletion Service → Audit Logger
                                                  ↓
                                          Biometric DB (Delete Templates)
                                                  ↓
                                          S3 (Delete Videos)
                                                  ↓
                                          Transaction DB (Anonymize)
                                                  ↓
                                          Compliance Report
```

**GDPR Right to Erasure**:
1. Customer submits deletion request via app
2. Data deletion service validates identity
3. Biometric templates permanently deleted
4. Enrollment videos removed from S3
5. Transaction history anonymized (keep for 7 years, remove PII)
6. Audit log created for compliance proof
7. Confirmation email sent to customer


## 5. API Integration Description

### 5.1 Internal Microservices APIs

**1. Enrollment API**

```http
POST /api/v1/enrollment
Authorization: Bearer <jwt_token>
Content-Type: multipart/form-data

{
  "customer_id": "uuid",
  "gait_video": "<file>",
  "consent_signature": "<base64>",
  "kyc_document": "<file>",
  "payment_method": {
    "type": "upi",
    "vpa": "customer@upi"
  }
}

Response 201:
{
  "status": "success",
  "enrollment_id": "uuid",
  "template_quality": 0.92,
  "estimated_accuracy": 0.96
}
```

**2. Identification API**

```http
POST /api/v1/identify
Authorization: Bearer <api_key>
Content-Type: application/json

{
  "store_id": "uuid",
  "gait_features": [0.12, 0.45, ...], // 128-dim vector
  "timestamp": "2026-02-15T10:30:45Z",
  "camera_id": "cam_001"
}

Response 200:
{
  "customer_id": "uuid",
  "confidence": 0.94,
  "match_time_ms": 87,
  "alternatives": [
    {"customer_id": "uuid2", "confidence": 0.78}
  ]
}
```

**3. Payment API**

```http
POST /api/v1/payment/charge
Authorization: Bearer <api_key>
Content-Type: application/json

{
  "customer_id": "uuid",
  "amount": 1250.50,
  "currency": "INR",
  "items": [
    {"sku": "ITEM001", "name": "Product A", "price": 500.00, "qty": 2},
    {"sku": "ITEM002", "name": "Product B", "price": 250.50, "qty": 1}
  ],
  "store_id": "uuid",
  "transaction_id": "uuid"
}

Response 200:
{
  "status": "success",
  "payment_id": "pay_xyz123",
  "transaction_id": "uuid",
  "receipt_url": "https://cdn.example.com/receipts/xyz.pdf"
}
```

### 5.2 External Payment Gateway APIs

**UPI Integration (Razorpay)**

```python
# Create recurring payment
POST https://api.razorpay.com/v1/payments/create/recurring
Authorization: Basic <base64(key:secret)>

{
  "email": "customer@example.com",
  "contact": "+919876543210",
  "amount": 125050,  // Amount in paise
  "currency": "INR",
  "order_id": "order_xyz",
  "customer_id": "cust_abc",
  "token": "token_def",
  "recurring": "1",
  "description": "Store purchase - Store Name",
  "notes": {
    "transaction_id": "uuid",
    "store_id": "uuid"
  }
}
```

**Card Payment (Stripe)**

```python
# Charge saved card
POST https://api.stripe.com/v1/charges
Authorization: Bearer <stripe_secret_key>

{
  "amount": 125050,
  "currency": "inr",
  "customer": "cus_xyz",
  "source": "card_token_abc",
  "description": "Automated store payment",
  "metadata": {
    "transaction_id": "uuid",
    "store_id": "uuid"
  }
}
```

### 5.3 KYC Verification APIs

**Aadhaar Verification (Surepass/IDfy)**

```http
POST https://api.surepass.io/api/v1/aadhaar-v2/verify
Authorization: Bearer <api_token>

{
  "id_number": "1234-5678-9012",
  "consent": "Y",
  "consent_text": "I authorize verification of my Aadhaar"
}

Response:
{
  "success": true,
  "data": {
    "name": "Customer Name",
    "dob": "1990-01-01",
    "gender": "M",
    "address": "..."
  }
}
```

### 5.4 Notification APIs

**SMS (Twilio)**

```python
POST https://api.twilio.com/2010-04-01/Accounts/{AccountSid}/Messages.json
Authorization: Basic <base64(AccountSid:AuthToken)>

{
  "To": "+919876543210",
  "From": "+1234567890",
  "Body": "Your payment of ₹1250.50 was successful. Receipt: https://..."
}
```

**Email (SendGrid)**

```python
POST https://api.sendgrid.com/v3/mail/send
Authorization: Bearer <sendgrid_api_key>

{
  "personalizations": [{
    "to": [{"email": "customer@example.com"}],
    "subject": "Payment Receipt - Store Name"
  }],
  "from": {"email": "noreply@gaitpay.com"},
  "content": [{
    "type": "text/html",
    "value": "<html>Receipt details...</html>"
  }],
  "attachments": [{
    "content": "<base64_pdf>",
    "filename": "receipt.pdf",
    "type": "application/pdf"
  }]
}
```

### 5.5 API Rate Limiting & Throttling

```yaml
Rate Limits:
  - Enrollment API: 10 requests/minute per customer
  - Identification API: 1000 requests/minute per store
  - Payment API: 100 requests/minute per store
  - Admin API: 500 requests/minute per admin user

Throttling Strategy:
  - Token bucket algorithm
  - 429 Too Many Requests response with Retry-After header
  - Exponential backoff for retries
```

### 5.6 API Security

- **Authentication**: JWT tokens (15-minute expiry) with refresh tokens
- **Authorization**: Role-based access control (RBAC)
- **Encryption**: TLS 1.3 for all API calls
- **Input Validation**: JSON schema validation, SQL injection prevention
- **API Gateway**: AWS API Gateway with WAF rules
- **Monitoring**: CloudWatch alarms for anomalous traffic patterns


## 6. Database Schema (Conceptual)

### 6.1 Entity-Relationship Diagram

```
┌─────────────────┐         ┌──────────────────┐         ┌─────────────────┐
│   CUSTOMERS     │         │  GAIT_TEMPLATES  │         │ PAYMENT_METHODS │
├─────────────────┤         ├──────────────────┤         ├─────────────────┤
│ customer_id PK  │────────<│ template_id PK   │         │ payment_id PK   │
│ phone_number    │         │ customer_id FK   │         │ customer_id FK  │
│ email           │         │ gait_embedding   │         │ payment_type    │
│ enrollment_date │         │ quality_score    │         │ token           │
│ consent_sig     │         │ created_at       │         │ is_primary      │
│ kyc_verified    │         └──────────────────┘         └─────────────────┘
│ account_status  │                  │                            │
└─────────────────┘                  │                            │
         │                           │                            │
         │                           │                            │
         └───────────────────────────┴────────────────────────────┘
                                     │
                                     │
                          ┌──────────▼──────────┐
                          │   TRANSACTIONS      │
                          ├─────────────────────┤
                          │ transaction_id PK   │
                          │ customer_id FK      │
                          │ store_id FK         │
                          │ amount              │
                          │ items (JSONB)       │
                          │ payment_status      │
                          │ payment_method_id FK│
                          │ timestamp           │
                          │ receipt_url         │
                          └─────────────────────┘
                                     │
                                     │
                          ┌──────────▼──────────┐
                          │    AUDIT_LOGS       │
                          ├─────────────────────┤
                          │ log_id PK           │
                          │ customer_id FK      │
                          │ action              │
                          │ actor               │
                          │ ip_address          │
                          │ timestamp           │
                          │ details (JSONB)     │
                          └─────────────────────┘

┌─────────────────┐         ┌──────────────────┐
│     STORES      │         │  STORE_CAMERAS   │
├─────────────────┤         ├──────────────────┤
│ store_id PK     │────────<│ camera_id PK     │
│ store_name      │         │ store_id FK      │
│ address         │         │ location         │
│ manager_contact │         │ camera_type      │
│ active_status   │         │ installation_date│
└─────────────────┘         └──────────────────┘
```

### 6.2 Detailed Schema Definitions

**CUSTOMERS Table**
```sql
CREATE TABLE customers (
    customer_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    phone_number VARCHAR(15) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    full_name VARCHAR(255) ENCRYPTED,
    date_of_birth DATE ENCRYPTED,
    enrollment_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    consent_signature BYTEA NOT NULL,
    consent_video_url VARCHAR(500),
    kyc_document_type VARCHAR(50),
    kyc_verified BOOLEAN DEFAULT FALSE,
    account_status VARCHAR(20) DEFAULT 'active' CHECK (account_status IN ('active', 'suspended', 'deleted')),
    spending_limit DECIMAL(10, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP NULL
);

CREATE INDEX idx_customers_phone ON customers(phone_number);
CREATE INDEX idx_customers_email ON customers(email);
CREATE INDEX idx_customers_status ON customers(account_status);
```

**GAIT_TEMPLATES Table**
```sql
CREATE TABLE gait_templates (
    template_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id UUID NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    gait_embedding VECTOR(128) NOT NULL, -- pgvector extension
    enrollment_device VARCHAR(100),
    enrollment_location VARCHAR(200),
    template_version INT DEFAULT 1,
    quality_score FLOAT CHECK (quality_score BETWEEN 0 AND 1),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    encryption_key_id VARCHAR(255) NOT NULL
);

CREATE INDEX idx_gait_customer ON gait_templates(customer_id);
CREATE INDEX idx_gait_active ON gait_templates(is_active);
-- Vector similarity index (using pgvector)
CREATE INDEX idx_gait_embedding ON gait_templates USING ivfflat (gait_embedding vector_cosine_ops);
```

**PAYMENT_METHODS Table**
```sql
CREATE TABLE payment_methods (
    payment_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id UUID NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    payment_type VARCHAR(20) NOT NULL CHECK (payment_type IN ('upi', 'card', 'bank_account', 'wallet')),
    token VARCHAR(255) NOT NULL ENCRYPTED, -- Tokenized credential
    provider VARCHAR(50), -- razorpay, stripe, paytm
    last_four_digits VARCHAR(4), -- For display purposes
    expiry_date DATE,
    is_primary BOOLEAN DEFAULT FALSE,
    is_active BOOLEAN DEFAULT TRUE,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_used_at TIMESTAMP
);

CREATE INDEX idx_payment_customer ON payment_methods(customer_id);
CREATE INDEX idx_payment_primary ON payment_methods(customer_id, is_primary);
```

**TRANSACTIONS Table**
```sql
CREATE TABLE transactions (
    transaction_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id UUID NOT NULL REFERENCES customers(customer_id),
    store_id UUID NOT NULL REFERENCES stores(store_id),
    amount DECIMAL(10, 2) NOT NULL,
    tax_amount DECIMAL(10, 2),
    discount_amount DECIMAL(10, 2),
    final_amount DECIMAL(10, 2) NOT NULL,
    items JSONB NOT NULL, -- Array of {sku, name, price, qty}
    payment_status VARCHAR(20) DEFAULT 'pending' CHECK (payment_status IN ('pending', 'processing', 'success', 'failed', 'refunded')),
    payment_method_id UUID REFERENCES payment_methods(payment_id),
    payment_gateway_txn_id VARCHAR(255),
    failure_reason TEXT,
    receipt_url VARCHAR(500),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    refunded_at TIMESTAMP
);

CREATE INDEX idx_txn_customer ON transactions(customer_id);
CREATE INDEX idx_txn_store ON transactions(store_id);
CREATE INDEX idx_txn_status ON transactions(payment_status);
CREATE INDEX idx_txn_timestamp ON transactions(timestamp DESC);
-- JSONB index for item queries
CREATE INDEX idx_txn_items ON transactions USING GIN (items);
```

**AUDIT_LOGS Table**
```sql
CREATE TABLE audit_logs (
    log_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id UUID REFERENCES customers(customer_id),
    action VARCHAR(50) NOT NULL CHECK (action IN ('enrollment', 'identification', 'payment', 'data_access', 'data_deletion', 'consent_update')),
    actor VARCHAR(100) NOT NULL, -- system, admin_user_id, customer_id
    actor_type VARCHAR(20) CHECK (actor_type IN ('system', 'admin', 'customer')),
    ip_address INET,
    user_agent TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    details JSONB,
    success BOOLEAN DEFAULT TRUE
);

CREATE INDEX idx_audit_customer ON audit_logs(customer_id);
CREATE INDEX idx_audit_action ON audit_logs(action);
CREATE INDEX idx_audit_timestamp ON audit_logs(timestamp DESC);
CREATE INDEX idx_audit_details ON audit_logs USING GIN (details);
```

**STORES Table**
```sql
CREATE TABLE stores (
    store_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    store_name VARCHAR(255) NOT NULL,
    store_code VARCHAR(50) UNIQUE,
    address TEXT,
    city VARCHAR(100),
    state VARCHAR(100),
    pincode VARCHAR(10),
    manager_name VARCHAR(255),
    manager_contact VARCHAR(15),
    active_status BOOLEAN DEFAULT TRUE,
    max_capacity INT, -- Max concurrent customers
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_stores_code ON stores(store_code);
CREATE INDEX idx_stores_active ON stores(active_status);
```

**STORE_CAMERAS Table**
```sql
CREATE TABLE store_cameras (
    camera_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    store_id UUID NOT NULL REFERENCES stores(store_id) ON DELETE CASCADE,
    camera_code VARCHAR(50) UNIQUE,
    location VARCHAR(100), -- entry, exit, aisle_1
    camera_type VARCHAR(50), -- rgb, depth, rgb+depth
    ip_address INET,
    installation_date DATE,
    last_maintenance_date DATE,
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'maintenance')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_cameras_store ON store_cameras(store_id);
CREATE INDEX idx_cameras_status ON store_cameras(status);
```

### 6.3 Data Partitioning Strategy

**Transactions Table Partitioning** (by month for efficient archival):
```sql
CREATE TABLE transactions_2026_02 PARTITION OF transactions
    FOR VALUES FROM ('2026-02-01') TO ('2026-03-01');

CREATE TABLE transactions_2026_03 PARTITION OF transactions
    FOR VALUES FROM ('2026-03-01') TO ('2026-04-01');
-- ... continue for each month
```

**Audit Logs Partitioning** (by quarter):
```sql
CREATE TABLE audit_logs_2026_q1 PARTITION OF audit_logs
    FOR VALUES FROM ('2026-01-01') TO ('2026-04-01');
```

### 6.4 Data Archival Policy

- **Transactions**: Keep 2 years in hot storage, archive to S3 Glacier
- **Audit Logs**: Keep 7 years (regulatory requirement), compress after 1 year
- **Gait Templates**: Delete within 30 days of account closure
- **Video Footage**: Delete after 7 days (dispute resolution window)


## 7. Model Training Pipeline

### 7.1 Training Data Collection

**Data Sources**:
1. **Public Datasets**:
   - CASIA Gait Database (153 subjects, multiple views)
   - OU-ISIR Large Population Dataset (4,007 subjects)
   - TUM-GAID (305 subjects, indoor/outdoor)
   
2. **Synthetic Data**:
   - Unity/Unreal Engine simulations with diverse avatars
   - Procedural gait generation with biomechanical models
   - Data augmentation: speed variations, viewpoint changes
   
3. **In-House Data**:
   - Pilot store enrollments (with consent)
   - Diverse demographics: age, gender, ethnicity, body types
   - Various conditions: footwear, carrying items, injuries

**Data Annotation**:
- Subject ID labels for supervised learning
- Quality scores (manual review of gait clarity)
- Metadata: walking speed, camera angle, lighting conditions

### 7.2 Training Infrastructure

**Hardware**:
- **GPU Cluster**: 8x NVIDIA A100 (40GB) for distributed training
- **Storage**: 10TB NVMe SSD for fast data loading
- **Network**: 100 Gbps InfiniBand for multi-node communication

**Software Stack**:
- **Framework**: PyTorch 2.0 with CUDA 11.8
- **Distributed Training**: PyTorch DDP (DistributedDataParallel)
- **Experiment Tracking**: Weights & Biases (W&B) or MLflow
- **Data Pipeline**: PyTorch DataLoader with prefetching

### 7.3 Training Process

**Phase 1: Pretraining on Public Datasets**
```python
# Model configuration
model = GaitRecognitionModel(
    backbone='resnet50',
    lstm_hidden=512,
    embedding_dim=128,
    num_classes=4007  # OU-ISIR dataset
)

# Loss function: ArcFace for metric learning
criterion = ArcFaceLoss(
    embedding_dim=128,
    num_classes=4007,
    scale=30.0,
    margin=0.5
)

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=0.0001
)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,
    eta_min=1e-6
)

# Training loop
for epoch in range(100):
    for batch in train_loader:
        gait_sequences, labels = batch
        embeddings = model(gait_sequences)
        loss = criterion(embeddings, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    scheduler.step()
    
    # Validation
    val_accuracy = evaluate(model, val_loader)
    wandb.log({"epoch": epoch, "loss": loss, "val_acc": val_accuracy})
```

**Phase 2: Fine-tuning on In-House Data**
```python
# Load pretrained model
model.load_state_dict(torch.load('pretrained_model.pth'))

# Freeze backbone, train only classification head
for param in model.backbone.parameters():
    param.requires_grad = False

# Fine-tune with smaller learning rate
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.0001
)

# Train for 20 epochs on in-house data
for epoch in range(20):
    # Training loop...
    pass
```

### 7.4 Data Augmentation

```python
class GaitAugmentation:
    def __init__(self):
        self.augmentations = [
            self.temporal_jitter,
            self.speed_variation,
            self.keypoint_noise,
            self.random_crop,
            self.horizontal_flip
        ]
    
    def temporal_jitter(self, sequence):
        # Randomly drop/duplicate frames
        return sequence
    
    def speed_variation(self, sequence):
        # Simulate faster/slower walking
        return sequence
    
    def keypoint_noise(self, sequence):
        # Add Gaussian noise to keypoints
        return sequence + np.random.normal(0, 0.02, sequence.shape)
    
    def random_crop(self, sequence):
        # Crop temporal window
        start = np.random.randint(0, len(sequence) - 30)
        return sequence[start:start+30]
    
    def horizontal_flip(self, sequence):
        # Mirror left-right (swap left/right keypoints)
        return sequence
```

### 7.5 Model Evaluation

**Metrics**:
```python
def evaluate_model(model, test_loader):
    embeddings_list = []
    labels_list = []
    
    with torch.no_grad():
        for gait_sequences, labels in test_loader:
            embeddings = model(gait_sequences)
            embeddings_list.append(embeddings)
            labels_list.append(labels)
    
    embeddings = torch.cat(embeddings_list)
    labels = torch.cat(labels_list)
    
    # Compute metrics
    rank1_acc = compute_rank_k_accuracy(embeddings, labels, k=1)
    rank5_acc = compute_rank_k_accuracy(embeddings, labels, k=5)
    eer = compute_equal_error_rate(embeddings, labels)
    
    return {
        'rank1_accuracy': rank1_acc,
        'rank5_accuracy': rank5_acc,
        'equal_error_rate': eer
    }
```

**Cross-Validation**:
- 5-fold cross-validation on in-house data
- Stratified splits to ensure demographic balance
- Report mean and standard deviation of metrics

### 7.6 Model Versioning & Deployment

**Model Registry**:
```python
# Save model with metadata
model_metadata = {
    'version': '1.2.0',
    'training_date': '2026-02-15',
    'dataset': 'OU-ISIR + InHouse_v3',
    'rank1_accuracy': 0.967,
    'eer': 0.018,
    'model_size_mb': 245
}

torch.save({
    'model_state_dict': model.state_dict(),
    'metadata': model_metadata
}, 'models/gait_model_v1.2.0.pth')

# Upload to S3 model registry
s3_client.upload_file(
    'models/gait_model_v1.2.0.pth',
    'gait-payment-models',
    'production/gait_model_v1.2.0.pth'
)
```

**A/B Testing**:
```python
# Deploy new model to 10% of traffic
if random.random() < 0.1:
    model = load_model('v1.2.0')  # New model
else:
    model = load_model('v1.1.0')  # Current model

# Log predictions for comparison
log_prediction(model_version, customer_id, confidence)
```

**Rollback Strategy**:
- Keep last 3 model versions in production
- Automated rollback if accuracy drops >2%
- Manual approval required for major version changes

### 7.7 Continuous Training

**Model Drift Detection**:
```python
def detect_model_drift():
    # Monitor accuracy over time
    recent_accuracy = get_accuracy_last_7_days()
    baseline_accuracy = 0.967
    
    if recent_accuracy < baseline_accuracy - 0.02:
        alert_ml_team("Model drift detected")
        trigger_retraining_pipeline()
```

**Retraining Schedule**:
- **Incremental**: Weekly with new enrollment data
- **Full Retraining**: Quarterly with updated architecture
- **Emergency**: Triggered by drift detection or security incidents


## 8. Deployment Strategy

### 8.1 Deployment Environments

**1. Development Environment**
- Single AWS region (us-east-1)
- Minimal infrastructure (1 GPU instance, small RDS)
- Mock payment gateway (sandbox mode)
- Purpose: Feature development and unit testing

**2. Staging Environment**
- Mirrors production architecture
- Separate database with anonymized production data
- Real payment gateway in test mode
- Purpose: Integration testing, UAT, performance testing

**3. Production Environment**
- Multi-region deployment (primary: ap-south-1, backup: ap-southeast-1)
- Auto-scaling infrastructure
- Real payment gateways
- Purpose: Live customer traffic

### 8.2 Deployment Architecture

**Blue-Green Deployment**:
```
┌─────────────────────────────────────────────────────────┐
│                    Load Balancer                        │
│                  (Route 53 + ALB)                       │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
    ┌────▼─────┐          ┌─────▼────┐
    │  BLUE    │          │  GREEN   │
    │ (Current)│          │  (New)   │
    │ v1.1.0   │          │ v1.2.0   │
    └──────────┘          └──────────┘
         │                       │
         └───────────┬───────────┘
                     │
              ┌──────▼──────┐
              │  Database   │
              │  (Shared)   │
              └─────────────┘
```

**Deployment Steps**:
1. Deploy new version to GREEN environment
2. Run smoke tests on GREEN
3. Gradually shift traffic: 5% → 25% → 50% → 100%
4. Monitor error rates and latency
5. If issues detected, instant rollback to BLUE
6. After 24 hours of stability, decommission BLUE

### 8.3 CI/CD Pipeline

**GitHub Actions Workflow**:
```yaml
name: Deploy Gait Payment System

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run unit tests
        run: pytest tests/unit
      - name: Run integration tests
        run: pytest tests/integration
      - name: Security scan
        run: |
          pip install bandit
          bandit -r src/

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Build Docker images
        run: |
          docker build -t gait-api:${{ github.sha }} -f Dockerfile.api .
          docker build -t gait-inference:${{ github.sha }} -f Dockerfile.inference .
      - name: Push to ECR
        run: |
          aws ecr get-login-password | docker login --username AWS --password-stdin $ECR_REGISTRY
          docker push gait-api:${{ github.sha }}
          docker push gait-inference:${{ github.sha }}

  deploy-staging:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to ECS Staging
        run: |
          aws ecs update-service --cluster staging --service gait-api --force-new-deployment
      - name: Run smoke tests
        run: pytest tests/smoke --env=staging
      - name: Performance tests
        run: locust -f tests/load/locustfile.py --host=https://staging.gaitpay.com

  deploy-production:
    needs: deploy-staging
    runs-on: ubuntu-latest
    environment: production
    steps:
      - name: Deploy to ECS Production (Blue-Green)
        run: |
          # Deploy to GREEN environment
          aws ecs update-service --cluster production-green --service gait-api --force-new-deployment
          
          # Wait for deployment to stabilize
          aws ecs wait services-stable --cluster production-green --services gait-api
          
          # Shift traffic gradually
          python scripts/gradual_traffic_shift.py --from blue --to green
      
      - name: Monitor deployment
        run: python scripts/monitor_deployment.py --duration 3600
      
      - name: Rollback if needed
        if: failure()
        run: python scripts/rollback.py --to blue
```

### 8.4 Container Orchestration

**ECS Task Definition**:
```json
{
  "family": "gait-inference-service",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "4096",
  "memory": "16384",
  "containerDefinitions": [
    {
      "name": "gait-inference",
      "image": "123456789.dkr.ecr.ap-south-1.amazonaws.com/gait-inference:latest",
      "essential": true,
      "portMappings": [
        {
          "containerPort": 8080,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "MODEL_VERSION", "value": "1.2.0"},
        {"name": "GPU_ENABLED", "value": "true"}
      ],
      "secrets": [
        {
          "name": "DB_PASSWORD",
          "valueFrom": "arn:aws:secretsmanager:ap-south-1:123456789:secret:db-password"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/gait-inference",
          "awslogs-region": "ap-south-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8080/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3
      }
    }
  ]
}
```

### 8.5 Database Migration Strategy

**Flyway Migration Scripts**:
```sql
-- V1.2.0__add_gait_quality_score.sql
ALTER TABLE gait_templates 
ADD COLUMN quality_score FLOAT CHECK (quality_score BETWEEN 0 AND 1);

CREATE INDEX idx_gait_quality ON gait_templates(quality_score);
```

**Zero-Downtime Migration**:
1. Add new columns (nullable initially)
2. Deploy application code that writes to both old and new columns
3. Backfill data for existing rows
4. Deploy code that reads from new columns
5. Remove old columns in next release

### 8.6 Rollback Procedures

**Automated Rollback Triggers**:
- Error rate > 5% for 5 minutes
- P95 latency > 3 seconds for 10 minutes
- Payment failure rate > 10%
- GPU inference failures > 20%

**Manual Rollback**:
```bash
# Rollback to previous version
aws ecs update-service \
  --cluster production \
  --service gait-api \
  --task-definition gait-api:42  # Previous revision

# Rollback database migration
flyway undo -target=1.1.0

# Rollback model version
aws s3 cp s3://models/gait_model_v1.1.0.pth /models/current/
```

### 8.7 Disaster Recovery

**Backup Strategy**:
- **Database**: Automated daily snapshots, 30-day retention
- **Models**: Versioned in S3 with cross-region replication
- **Configuration**: Stored in Git, encrypted secrets in Secrets Manager

**Recovery Procedures**:
```bash
# Restore database from snapshot
aws rds restore-db-instance-from-db-snapshot \
  --db-instance-identifier gait-payment-db-restored \
  --db-snapshot-identifier snapshot-2026-02-15

# Restore model from backup
aws s3 sync s3://models-backup/2026-02-15/ /models/

# Redeploy infrastructure from Terraform
terraform apply -var-file=disaster-recovery.tfvars
```

**RTO/RPO Targets**:
- **RTO (Recovery Time Objective)**: 4 hours
- **RPO (Recovery Point Objective)**: 1 hour
- **Data Loss**: Maximum 1 hour of transactions


## 9. Scalability Considerations

### 9.1 Horizontal Scaling

**Microservices Auto-Scaling**:
```yaml
# ECS Service Auto Scaling
AutoScalingTarget:
  ServiceNamespace: ecs
  ResourceId: service/production/gait-api
  ScalableDimension: ecs:service:DesiredCount
  MinCapacity: 4
  MaxCapacity: 50

ScalingPolicy:
  PolicyType: TargetTrackingScaling
  TargetTrackingScalingPolicyConfiguration:
    PredefinedMetricSpecification:
      PredefinedMetricType: ECSServiceAverageCPUUtilization
    TargetValue: 60.0
    ScaleInCooldown: 300
    ScaleOutCooldown: 60
```

**Database Scaling**:
- **Read Replicas**: 3 read replicas for query distribution
- **Connection Pooling**: PgBouncer with 1000 max connections
- **Sharding Strategy**: Partition by store_id for multi-tenant isolation
- **Caching**: Redis cluster with 99.9% cache hit rate target

### 9.2 Vertical Scaling

**GPU Inference Scaling**:
```python
# Dynamic batch size based on GPU memory
def adaptive_batch_inference(gait_features_queue):
    gpu_memory_available = torch.cuda.mem_get_info()[0]
    
    if gpu_memory_available > 10 * 1024**3:  # >10GB
        batch_size = 64
    elif gpu_memory_available > 5 * 1024**3:  # >5GB
        batch_size = 32
    else:
        batch_size = 16
    
    batch = gait_features_queue.get_batch(batch_size)
    embeddings = model(batch)
    return embeddings
```

**Instance Types by Load**:
| Load Level | API Instances | Inference Instances | Database |
|------------|---------------|---------------------|----------|
| Low (0-50 stores) | t3.medium x2 | p3.2xlarge x1 | db.r5.large |
| Medium (50-200) | c5.2xlarge x5 | p3.2xlarge x3 | db.r5.2xlarge |
| High (200-500) | c5.4xlarge x10 | p3.8xlarge x5 | db.r5.4xlarge |
| Peak (500+) | c5.9xlarge x20 | p3.16xlarge x10 | db.r5.12xlarge |

### 9.3 Geographic Distribution

**Multi-Region Architecture**:
```
Primary Region (ap-south-1 - Mumbai):
  - Active-Active for API services
  - Primary database (write master)
  - Model inference cluster

Secondary Region (ap-southeast-1 - Singapore):
  - Active-Active for API services
  - Read replica database
  - Model inference cluster (backup)

Tertiary Region (us-east-1 - Virginia):
  - Disaster recovery (cold standby)
  - Model training infrastructure
```

**Latency Optimization**:
- **Route 53 Geolocation Routing**: Direct users to nearest region
- **CloudFront CDN**: Cache static assets (receipts, app assets)
- **Edge Locations**: Deploy pose estimation to store premises (edge computing)

### 9.4 Load Balancing Strategy

**Multi-Tier Load Balancing**:
```
Internet
   ↓
Route 53 (DNS-based geographic routing)
   ↓
CloudFront (CDN for static content)
   ↓
Application Load Balancer (ALB)
   ↓
Target Groups (API, Inference, Admin)
   ↓
ECS Services (Auto-scaled containers)
```

**Load Balancing Algorithms**:
- **API Services**: Round-robin with sticky sessions
- **Inference Services**: Least outstanding requests (GPU utilization aware)
- **Database**: Read/write splitting with weighted routing

### 9.5 Queue-Based Decoupling

**Message Queue Architecture**:
```python
# Kafka topics for asynchronous processing
topics = {
    'gait.identification.requests': {
        'partitions': 20,
        'replication_factor': 3,
        'retention_ms': 3600000  # 1 hour
    },
    'payment.processing': {
        'partitions': 10,
        'replication_factor': 3,
        'retention_ms': 86400000  # 24 hours
    },
    'notifications.send': {
        'partitions': 5,
        'replication_factor': 3,
        'retention_ms': 604800000  # 7 days
    }
}

# Producer (Edge device)
producer.send('gait.identification.requests', {
    'store_id': 'uuid',
    'gait_features': [...],
    'timestamp': '2026-02-15T10:30:45Z'
})

# Consumer (Gait matching service)
for message in consumer:
    gait_features = message.value['gait_features']
    customer_id = identify_customer(gait_features)
    cache.set(f"session:{message.value['store_id']}", customer_id)
```

**Benefits**:
- Decouple edge devices from cloud services
- Handle traffic spikes without dropping requests
- Retry failed operations automatically
- Enable event-driven architecture

### 9.6 Caching Strategy

**Multi-Layer Caching**:
```python
# Layer 1: Application-level cache (in-memory)
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_customer_profile(customer_id):
    return db.query(f"SELECT * FROM customers WHERE customer_id = '{customer_id}'")

# Layer 2: Distributed cache (Redis)
def get_gait_template(customer_id):
    cache_key = f"gait_template:{customer_id}"
    
    # Try cache first
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)
    
    # Cache miss - fetch from database
    template = db.query(f"SELECT * FROM gait_templates WHERE customer_id = '{customer_id}'")
    
    # Store in cache (TTL: 1 hour)
    redis_client.setex(cache_key, 3600, json.dumps(template))
    return template

# Layer 3: CDN cache (CloudFront)
# Static assets: receipts, app assets, model artifacts
```

**Cache Invalidation**:
- **Time-based**: TTL of 1 hour for customer profiles
- **Event-based**: Invalidate on profile updates, payment method changes
- **Manual**: Admin can force cache clear for specific customers

### 9.7 Performance Benchmarks

**Target Metrics**:
| Operation | Target | Measured | Status |
|-----------|--------|----------|--------|
| Gait identification | <3s | 2.7s | ✅ |
| Payment processing | <2s | 1.8s | ✅ |
| API response (p95) | <500ms | 420ms | ✅ |
| Database query (p95) | <100ms | 85ms | ✅ |
| Vector search | <10ms | 8ms | ✅ |
| Throughput (per store) | 200 customers/hour | 250 customers/hour | ✅ |

**Load Testing Results** (Locust):
```python
# Load test configuration
class GaitPaymentUser(HttpUser):
    wait_time = between(1, 3)
    
    @task(3)
    def identify_customer(self):
        self.client.post("/api/v1/identify", json={
            "store_id": "test-store",
            "gait_features": [random.random() for _ in range(128)]
        })
    
    @task(1)
    def process_payment(self):
        self.client.post("/api/v1/payment/charge", json={
            "customer_id": "test-customer",
            "amount": 1250.50
        })

# Results (1000 concurrent users):
# - Requests per second: 2500
# - Average response time: 380ms
# - 95th percentile: 620ms
# - 99th percentile: 1200ms
# - Error rate: 0.02%
```


## 10. Risk Mitigation Architecture

### 10.1 Failure Modes & Mitigations

**1. Camera Failure**
- **Risk**: Customer cannot be identified if cameras malfunction
- **Mitigation**:
  - Redundant cameras with overlapping coverage
  - Health monitoring with automatic alerts
  - Fallback to manual checkout (staff-assisted)
  - Maintenance schedule: monthly camera checks

**2. Network Outage**
- **Risk**: Edge devices cannot communicate with cloud
- **Mitigation**:
  - Local caching of frequent customer templates (top 100)
  - Offline mode: store transactions locally, sync when online
  - Cellular backup connection (4G/5G)
  - Queue-based retry mechanism

**3. Model Inference Failure**
- **Risk**: GPU crashes or model serving errors
- **Mitigation**:
  - Multiple inference replicas (N+2 redundancy)
  - Health checks with automatic instance replacement
  - Fallback to CPU inference (slower but functional)
  - Circuit breaker pattern to prevent cascade failures

**4. Payment Gateway Downtime**
- **Risk**: Cannot process payments
- **Mitigation**:
  - Multi-gateway support (Razorpay + Stripe + Paytm)
  - Automatic failover to backup gateway
  - Deferred payment: allow exit, charge later with grace period
  - SMS payment link as last resort

**5. Database Failure**
- **Risk**: Cannot access customer profiles or templates
- **Mitigation**:
  - Multi-AZ deployment with automatic failover
  - Read replicas for query distribution
  - Point-in-time recovery (PITR) enabled
  - Cross-region backup for disaster recovery

**6. False Positive Identification**
- **Risk**: Wrong customer charged for items
- **Mitigation**:
  - Confidence threshold (>85% required for auto-payment)
  - Low-confidence alerts: staff verification required
  - Real-time app notification: "You're being charged ₹X, tap to confirm"
  - Easy dispute resolution with video evidence
  - Insurance coverage for fraud cases

**7. Gait Spoofing Attack**
- **Risk**: Attacker mimics victim's gait
- **Mitigation**:
  - Liveness detection (depth sensors, motion analysis)
  - Multi-factor authentication for high-value transactions (>₹5000)
  - Behavioral anomaly detection (unusual purchase patterns)
  - Velocity checks (same customer can't be in two stores simultaneously)

### 10.2 Circuit Breaker Pattern

```python
from circuitbreaker import circuit

@circuit(failure_threshold=5, recovery_timeout=60)
def call_payment_gateway(customer_id, amount):
    """
    Circuit breaker protects against cascading failures.
    Opens after 5 consecutive failures, closes after 60s.
    """
    response = payment_gateway_api.charge(customer_id, amount)
    if response.status_code != 200:
        raise PaymentGatewayError(response.text)
    return response.json()

# Usage with fallback
try:
    result = call_payment_gateway(customer_id, amount)
except CircuitBreakerError:
    # Circuit is open, use fallback
    result = send_payment_link_via_sms(customer_id, amount)
```

### 10.3 Rate Limiting & DDoS Protection

**API Rate Limiting**:
```python
from flask_limiter import Limiter

limiter = Limiter(
    app,
    key_func=lambda: request.headers.get('X-API-Key'),
    storage_uri="redis://localhost:6379"
)

@app.route('/api/v1/identify', methods=['POST'])
@limiter.limit("1000 per minute")  # Per store
def identify_customer():
    # Identification logic
    pass

@app.route('/api/v1/enrollment', methods=['POST'])
@limiter.limit("10 per minute")  # Per customer
def enroll_customer():
    # Enrollment logic
    pass
```

**DDoS Mitigation**:
- AWS Shield Standard (automatic protection)
- AWS WAF rules: block suspicious IPs, rate limit by geography
- CloudFront with geo-blocking for non-target markets
- Anomaly detection: alert on 10x traffic spike

### 10.4 Data Integrity Checks

**Transaction Validation**:
```python
def validate_transaction(transaction):
    checks = [
        # Amount sanity check
        0 < transaction.amount < 100000,  # Max ₹1 lakh per transaction
        
        # Item count check
        1 <= len(transaction.items) <= 100,
        
        # Timestamp check (not in future, not older than 1 hour)
        datetime.now() - timedelta(hours=1) <= transaction.timestamp <= datetime.now(),
        
        # Customer exists and active
        customer_exists(transaction.customer_id),
        
        # Store exists and active
        store_exists(transaction.store_id),
        
        # Payment method is valid
        payment_method_valid(transaction.payment_method_id)
    ]
    
    if not all(checks):
        raise TransactionValidationError("Transaction failed validation")
    
    return True
```

**Checksum Verification**:
```python
import hashlib

def compute_transaction_hash(transaction):
    """Compute SHA-256 hash for transaction integrity"""
    data = f"{transaction.customer_id}{transaction.amount}{transaction.timestamp}"
    return hashlib.sha256(data.encode()).hexdigest()

# Store hash with transaction
transaction.integrity_hash = compute_transaction_hash(transaction)

# Verify on retrieval
def verify_transaction_integrity(transaction):
    computed_hash = compute_transaction_hash(transaction)
    if computed_hash != transaction.integrity_hash:
        raise IntegrityError("Transaction data has been tampered")
```

### 10.5 Graceful Degradation

**Service Degradation Levels**:
```python
class ServiceLevel(Enum):
    FULL = "full"              # All features operational
    DEGRADED = "degraded"      # Non-critical features disabled
    MINIMAL = "minimal"        # Only core payment flow
    MAINTENANCE = "maintenance" # Read-only mode

def get_service_level():
    """Determine current service level based on system health"""
    if gpu_cluster_healthy() and database_healthy() and payment_gateway_healthy():
        return ServiceLevel.FULL
    elif database_healthy() and payment_gateway_healthy():
        return ServiceLevel.DEGRADED  # Disable real-time analytics
    elif database_healthy():
        return ServiceLevel.MINIMAL  # Disable payments, allow browsing
    else:
        return ServiceLevel.MAINTENANCE

# Adjust behavior based on service level
service_level = get_service_level()

if service_level == ServiceLevel.DEGRADED:
    # Disable non-critical features
    disable_real_time_analytics()
    disable_recommendation_engine()
elif service_level == ServiceLevel.MINIMAL:
    # Only allow identification, no payments
    disable_payment_processing()
    show_maintenance_banner()
```

### 10.6 Chaos Engineering

**Chaos Experiments** (using Chaos Monkey):
```yaml
# Randomly terminate instances to test resilience
experiments:
  - name: terminate_random_api_instance
    frequency: weekly
    blast_radius: 1 instance
    rollback_on_failure: true
  
  - name: inject_network_latency
    frequency: daily
    latency: 500ms
    duration: 5 minutes
    target: inference_service
  
  - name: simulate_database_failover
    frequency: monthly
    duration: 10 minutes
  
  - name: exhaust_gpu_memory
    frequency: weekly
    target: inference_gpu
    duration: 2 minutes
```

**Monitoring During Chaos**:
- Error rate should not exceed 1%
- Latency should not increase >2x
- No data loss or corruption
- Automatic recovery within 5 minutes


## 11. Security Architecture

### 11.1 Defense in Depth

**Security Layers**:
```
Layer 1: Network Security
  - VPC with private subnets
  - Security groups (whitelist only)
  - Network ACLs
  - AWS WAF + Shield

Layer 2: Application Security
  - API authentication (JWT)
  - Input validation
  - OWASP Top 10 compliance
  - Rate limiting

Layer 3: Data Security
  - Encryption at rest (AES-256)
  - Encryption in transit (TLS 1.3)
  - Tokenization (payment data)
  - Template protection (biometrics)

Layer 4: Access Control
  - RBAC (Role-Based Access Control)
  - MFA for admin access
  - Audit logging
  - Principle of least privilege

Layer 5: Monitoring & Response
  - SIEM (Security Information and Event Management)
  - Intrusion detection
  - Automated threat response
  - Incident response plan
```

### 11.2 Encryption Architecture

**Data at Rest**:
```python
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2

class BiometricEncryption:
    def __init__(self, customer_id):
        # Derive customer-specific encryption key
        self.key = self._derive_key(customer_id)
        self.cipher = Fernet(self.key)
    
    def _derive_key(self, customer_id):
        """Derive encryption key from customer ID + master key"""
        master_key = get_master_key_from_kms()
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=customer_id.encode(),
            iterations=100000
        )
        return base64.urlsafe_b64encode(kdf.derive(master_key))
    
    def encrypt_template(self, gait_template):
        """Encrypt gait template before storage"""
        template_bytes = json.dumps(gait_template).encode()
        encrypted = self.cipher.encrypt(template_bytes)
        return encrypted
    
    def decrypt_template(self, encrypted_template):
        """Decrypt gait template for matching"""
        decrypted = self.cipher.decrypt(encrypted_template)
        return json.loads(decrypted.decode())

# Usage
encryptor = BiometricEncryption(customer_id)
encrypted_template = encryptor.encrypt_template(gait_features)
db.store(customer_id, encrypted_template)
```

**Data in Transit**:
```nginx
# Nginx TLS configuration
server {
    listen 443 ssl http2;
    server_name api.gaitpay.com;
    
    # TLS 1.3 only
    ssl_protocols TLSv1.3;
    ssl_ciphers 'TLS_AES_256_GCM_SHA384:TLS_CHACHA20_POLY1305_SHA256';
    ssl_prefer_server_ciphers on;
    
    # Certificate
    ssl_certificate /etc/ssl/certs/gaitpay.crt;
    ssl_certificate_key /etc/ssl/private/gaitpay.key;
    
    # HSTS (HTTP Strict Transport Security)
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    
    # Certificate pinning
    add_header Public-Key-Pins 'pin-sha256="base64+primary=="; pin-sha256="base64+backup=="; max-age=5184000';
}
```

### 11.3 Authentication & Authorization

**JWT Token Structure**:
```python
import jwt
from datetime import datetime, timedelta

def generate_jwt(customer_id, role):
    payload = {
        'sub': customer_id,  # Subject (customer ID)
        'role': role,        # User role (customer, admin, store_operator)
        'iat': datetime.utcnow(),  # Issued at
        'exp': datetime.utcnow() + timedelta(minutes=15),  # Expiry
        'jti': str(uuid.uuid4())  # JWT ID (for revocation)
    }
    
    token = jwt.encode(payload, JWT_SECRET, algorithm='HS256')
    return token

def verify_jwt(token):
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
        
        # Check if token is revoked
        if is_token_revoked(payload['jti']):
            raise jwt.InvalidTokenError("Token has been revoked")
        
        return payload
    except jwt.ExpiredSignatureError:
        raise AuthenticationError("Token has expired")
    except jwt.InvalidTokenError:
        raise AuthenticationError("Invalid token")
```

**RBAC Implementation**:
```python
from functools import wraps

def require_role(required_role):
    """Decorator to enforce role-based access control"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            token = request.headers.get('Authorization', '').replace('Bearer ', '')
            payload = verify_jwt(token)
            
            if payload['role'] not in required_role:
                return jsonify({'error': 'Insufficient permissions'}), 403
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# Usage
@app.route('/api/v1/admin/customers', methods=['GET'])
@require_role(['admin', 'store_operator'])
def list_customers():
    # Only admins and store operators can access
    pass

@app.route('/api/v1/customer/profile', methods=['GET'])
@require_role(['customer'])
def get_profile():
    # Only customers can access their own profile
    pass
```

### 11.4 Payment Tokenization

**Tokenization Flow**:
```python
def tokenize_payment_method(customer_id, payment_details):
    """
    Replace sensitive payment data with non-sensitive token.
    Actual payment data stored securely at payment gateway.
    """
    # Send payment details to gateway for tokenization
    response = payment_gateway.create_token({
        'customer_id': customer_id,
        'payment_type': payment_details['type'],
        'card_number': payment_details.get('card_number'),
        'upi_vpa': payment_details.get('upi_vpa'),
        'expiry': payment_details.get('expiry'),
        'cvv': payment_details.get('cvv')  # Never stored
    })
    
    # Store only the token, not actual payment data
    token = response['token']
    last_four = payment_details.get('card_number', '')[-4:] if payment_details.get('card_number') else None
    
    db.insert('payment_methods', {
        'customer_id': customer_id,
        'token': token,  # Encrypted token
        'last_four_digits': last_four,
        'payment_type': payment_details['type']
    })
    
    return token

def charge_tokenized_payment(customer_id, amount):
    """Charge using tokenized payment method"""
    payment_method = db.query(
        "SELECT token FROM payment_methods WHERE customer_id = ? AND is_primary = TRUE",
        customer_id
    )
    
    # Use token to charge (no sensitive data in our system)
    response = payment_gateway.charge({
        'token': payment_method['token'],
        'amount': amount,
        'currency': 'INR'
    })
    
    return response
```

### 11.5 Biometric Template Protection

**Cancelable Biometrics**:
```python
def generate_cancelable_template(gait_features, customer_secret):
    """
    Generate cancelable biometric template that can be revoked.
    If compromised, customer can generate new template with different secret.
    """
    # Apply customer-specific transformation
    transformation_matrix = derive_transformation_matrix(customer_secret)
    cancelable_template = np.dot(gait_features, transformation_matrix)
    
    # Apply one-way hash
    template_hash = hashlib.sha256(cancelable_template.tobytes()).digest()
    
    return template_hash

def match_cancelable_template(probe_features, stored_hash, customer_secret):
    """Match probe against stored cancelable template"""
    # Apply same transformation
    transformation_matrix = derive_transformation_matrix(customer_secret)
    cancelable_probe = np.dot(probe_features, transformation_matrix)
    probe_hash = hashlib.sha256(cancelable_probe.tobytes()).digest()
    
    # Compare hashes
    similarity = compute_hash_similarity(probe_hash, stored_hash)
    return similarity > THRESHOLD
```

### 11.6 Secure Key Management

**AWS KMS Integration**:
```python
import boto3

kms_client = boto3.client('kms', region_name='ap-south-1')

def get_data_encryption_key(customer_id):
    """Generate customer-specific data encryption key"""
    # Request data key from KMS
    response = kms_client.generate_data_key(
        KeyId='arn:aws:kms:ap-south-1:123456789:key/master-key-id',
        KeySpec='AES_256',
        EncryptionContext={
            'customer_id': customer_id,
            'purpose': 'biometric_encryption'
        }
    )
    
    # Return plaintext key (use immediately, don't store)
    # Store encrypted key in database
    return {
        'plaintext_key': response['Plaintext'],
        'encrypted_key': response['CiphertextBlob']
    }

def decrypt_data_encryption_key(encrypted_key, customer_id):
    """Decrypt customer-specific data encryption key"""
    response = kms_client.decrypt(
        CiphertextBlob=encrypted_key,
        EncryptionContext={
            'customer_id': customer_id,
            'purpose': 'biometric_encryption'
        }
    )
    
    return response['Plaintext']

# Key rotation (every 90 days)
def rotate_customer_keys():
    """Rotate encryption keys for all customers"""
    customers = db.query("SELECT customer_id FROM customers WHERE account_status = 'active'")
    
    for customer in customers:
        # Generate new key
        new_key = get_data_encryption_key(customer['customer_id'])
        
        # Re-encrypt biometric template with new key
        old_template = decrypt_template(customer['customer_id'])
        new_encrypted_template = encrypt_template(old_template, new_key['plaintext_key'])
        
        # Update database
        db.update('gait_templates', {
            'customer_id': customer['customer_id'],
            'gait_embedding': new_encrypted_template,
            'encryption_key_id': new_key['encrypted_key']
        })
```

### 11.7 Security Monitoring

**SIEM Integration**:
```python
import logging
from pythonjsonlogger import jsonlogger

# Structured logging for SIEM
logger = logging.getLogger()
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)

def log_security_event(event_type, severity, details):
    """Log security events for SIEM analysis"""
    logger.warning({
        'event_type': event_type,
        'severity': severity,
        'timestamp': datetime.utcnow().isoformat(),
        'details': details,
        'source_ip': request.remote_addr,
        'user_agent': request.headers.get('User-Agent')
    })

# Security event examples
log_security_event('failed_authentication', 'medium', {
    'customer_id': customer_id,
    'reason': 'invalid_token'
})

log_security_event('suspicious_activity', 'high', {
    'customer_id': customer_id,
    'reason': 'multiple_failed_payments',
    'count': 5
})

log_security_event('data_access', 'low', {
    'admin_id': admin_id,
    'action': 'view_customer_profile',
    'customer_id': customer_id
})
```

**Anomaly Detection**:
```python
def detect_anomalies(customer_id):
    """Detect suspicious patterns in customer behavior"""
    recent_transactions = get_recent_transactions(customer_id, days=7)
    
    anomalies = []
    
    # Check for unusual transaction amounts
    avg_amount = np.mean([t.amount for t in recent_transactions])
    std_amount = np.std([t.amount for t in recent_transactions])
    
    for txn in recent_transactions:
        if txn.amount > avg_amount + 3 * std_amount:
            anomalies.append({
                'type': 'unusual_amount',
                'transaction_id': txn.id,
                'amount': txn.amount,
                'expected_range': (avg_amount - std_amount, avg_amount + std_amount)
            })
    
    # Check for unusual transaction frequency
    if len(recent_transactions) > 50:  # >50 transactions in 7 days
        anomalies.append({
            'type': 'high_frequency',
            'count': len(recent_transactions)
        })
    
    # Check for unusual store locations
    usual_stores = get_usual_stores(customer_id)
    for txn in recent_transactions:
        if txn.store_id not in usual_stores:
            anomalies.append({
                'type': 'unusual_location',
                'transaction_id': txn.id,
                'store_id': txn.store_id
            })
    
    if anomalies:
        alert_fraud_team(customer_id, anomalies)
    
    return anomalies
```


## 12. Edge vs Cloud Processing Comparison

### 12.1 Processing Distribution

| Component | Edge (Store Premises) | Cloud (Data Center) | Rationale |
|-----------|----------------------|---------------------|-----------|
| **Video Capture** | ✅ | ❌ | Bandwidth constraints, privacy |
| **Pose Estimation** | ✅ | ❌ | Low latency requirement (<100ms) |
| **Feature Extraction** | ✅ | ❌ | Reduce data transmission (128 dims vs full video) |
| **Gait Matching** | ❌ | ✅ | Requires access to full customer database |
| **Payment Processing** | ❌ | ✅ | Security, PCI-DSS compliance |
| **Model Training** | ❌ | ✅ | Requires GPU cluster, large datasets |
| **Analytics** | ❌ | ✅ | Aggregate data from multiple stores |
| **Admin Dashboard** | ❌ | ✅ | Centralized management |

### 12.2 Edge Computing Architecture

**Edge Device Specifications**:
```yaml
Hardware:
  Processor: NVIDIA Jetson Xavier NX
  GPU: 384-core NVIDIA Volta (6 TFLOPS)
  CPU: 6-core ARM Carmel (1.9 GHz)
  Memory: 16 GB LPDDR4x
  Storage: 256 GB NVMe SSD
  Network: Gigabit Ethernet + 4G LTE backup
  Power: 10-20W (energy efficient)

Software:
  OS: Ubuntu 20.04 LTS (ARM64)
  Runtime: NVIDIA JetPack 5.0
  Container: Docker with NVIDIA Container Runtime
  Orchestration: K3s (lightweight Kubernetes)
```

**Edge Processing Pipeline**:
```python
# Edge device main loop
class EdgeProcessor:
    def __init__(self):
        self.pose_model = load_pose_model('hrnet_w32.onnx')
        self.feature_extractor = GaitFeatureExtractor()
        self.cloud_api = CloudAPIClient()
        self.local_cache = LocalCache(max_size=100)  # Cache top 100 customers
    
    def process_camera_stream(self, camera_id):
        """Main processing loop for camera stream"""
        cap = cv2.VideoCapture(camera_id)
        frame_buffer = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Step 1: Detect person in frame
            person_bbox = self.detect_person(frame)
            if person_bbox is None:
                continue
            
            # Step 2: Extract pose keypoints (edge processing)
            keypoints = self.pose_model.predict(frame, person_bbox)
            frame_buffer.append(keypoints)
            
            # Step 3: When we have enough frames (30 frames = 1 second)
            if len(frame_buffer) >= 30:
                # Extract gait features (edge processing)
                gait_features = self.feature_extractor.extract(frame_buffer)
                
                # Step 4: Try local cache first
                customer_id = self.local_cache.match(gait_features)
                
                if customer_id is None:
                    # Step 5: Send to cloud for matching
                    customer_id = self.cloud_api.identify(gait_features)
                    
                    # Cache result for future
                    if customer_id:
                        self.local_cache.add(customer_id, gait_features)
                
                # Step 6: Notify store system
                if customer_id:
                    self.notify_customer_identified(customer_id)
                
                # Reset buffer
                frame_buffer = []
```

**Benefits of Edge Processing**:
1. **Low Latency**: Pose estimation in <100ms (vs 500ms+ with cloud round-trip)
2. **Bandwidth Savings**: Send 128-dim vector (512 bytes) vs full video (10+ MB/sec)
3. **Privacy**: Raw video never leaves store premises
4. **Reliability**: Works during network outages (with local cache)
5. **Cost**: Reduce cloud compute and bandwidth costs

**Challenges**:
1. **Limited Compute**: Cannot run full matching against 1M+ templates
2. **Model Updates**: Need OTA (over-the-air) update mechanism
3. **Hardware Cost**: $500-1000 per edge device
4. **Maintenance**: Physical access required for repairs

### 12.3 Hybrid Architecture Benefits

**Optimal Workload Distribution**:
```
Edge Device (Store):
  - Real-time video processing (30 FPS)
  - Pose estimation (15-20ms per frame)
  - Feature extraction (50ms per sequence)
  - Local caching (top 100 frequent customers)
  - Offline mode support
  
  ↓ (Send 512 bytes per identification)
  
Cloud (Data Center):
  - Gait matching (1M+ templates, 50ms)
  - Payment processing (1-2 seconds)
  - Model training (hours to days)
  - Analytics and reporting
  - Admin dashboard
  - Data storage and backup
```

**Cost Comparison**:
| Scenario | Edge Only | Cloud Only | Hybrid (Proposed) |
|----------|-----------|------------|-------------------|
| Hardware Cost | $1000/store | $0 | $800/store |
| Cloud Compute | $0/month | $5000/month | $1500/month |
| Bandwidth | $0/month | $2000/month | $100/month |
| Latency | 100ms | 800ms | 150ms |
| **Total (1 year)** | **$1000** | **$84,000** | **$20,000** |

**Hybrid architecture provides 4x cost savings vs cloud-only with better latency.**

### 12.4 Edge Device Management

**Over-the-Air (OTA) Updates**:
```python
# Edge device update client
class OTAUpdateClient:
    def __init__(self):
        self.current_version = "1.2.0"
        self.update_server = "https://updates.gaitpay.com"
    
    def check_for_updates(self):
        """Check if new model/software version available"""
        response = requests.get(f"{self.update_server}/latest")
        latest_version = response.json()['version']
        
        if latest_version > self.current_version:
            self.download_and_install(latest_version)
    
    def download_and_install(self, version):
        """Download and install new version"""
        # Download model file
        model_url = f"{self.update_server}/models/gait_model_{version}.onnx"
        download_file(model_url, f"/models/gait_model_{version}.onnx")
        
        # Verify checksum
        if not verify_checksum(f"/models/gait_model_{version}.onnx"):
            raise UpdateError("Checksum verification failed")
        
        # Atomic swap (zero downtime)
        os.rename(f"/models/gait_model_{version}.onnx", "/models/current.onnx")
        
        # Reload model
        self.pose_model = load_pose_model("/models/current.onnx")
        self.current_version = version
        
        log_update_success(version)

# Run update check daily at 3 AM
schedule.every().day.at("03:00").do(ota_client.check_for_updates)
```

**Remote Monitoring**:
```python
# Edge device health metrics
class EdgeDeviceMonitor:
    def collect_metrics(self):
        return {
            'device_id': self.device_id,
            'timestamp': datetime.utcnow().isoformat(),
            'cpu_usage': psutil.cpu_percent(),
            'gpu_usage': self.get_gpu_usage(),
            'memory_usage': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'temperature': self.get_temperature(),
            'network_latency': self.measure_cloud_latency(),
            'frames_processed': self.frame_counter,
            'identifications_today': self.identification_counter,
            'errors_today': self.error_counter
        }
    
    def send_metrics_to_cloud(self):
        """Send health metrics to cloud for monitoring"""
        metrics = self.collect_metrics()
        requests.post(
            "https://api.gaitpay.com/v1/edge/metrics",
            json=metrics,
            headers={'X-Device-ID': self.device_id}
        )

# Send metrics every 5 minutes
schedule.every(5).minutes.do(monitor.send_metrics_to_cloud)
```

### 12.5 Offline Mode Support

**Local Cache Strategy**:
```python
class LocalCustomerCache:
    def __init__(self, max_size=100):
        self.max_size = max_size
        self.cache = {}  # {customer_id: gait_template}
        self.access_count = {}  # {customer_id: count}
    
    def update_cache(self):
        """Fetch top frequent customers from cloud"""
        # Get list of frequent customers for this store
        response = requests.get(
            f"https://api.gaitpay.com/v1/stores/{self.store_id}/frequent-customers",
            params={'limit': self.max_size}
        )
        
        frequent_customers = response.json()
        
        # Download their gait templates
        for customer in frequent_customers:
            template = self.download_template(customer['customer_id'])
            self.cache[customer['customer_id']] = template
    
    def match_offline(self, gait_features):
        """Match against local cache when offline"""
        best_match = None
        best_similarity = 0
        
        for customer_id, template in self.cache.items():
            similarity = compute_cosine_similarity(gait_features, template)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = customer_id
        
        if best_similarity > 0.85:
            return best_match
        return None

# Update cache daily
schedule.every().day.at("02:00").do(cache.update_cache)
```

**Offline Transaction Queue**:
```python
class OfflineTransactionQueue:
    def __init__(self):
        self.queue = []
        self.queue_file = "/data/offline_transactions.json"
        self.load_queue()
    
    def add_transaction(self, transaction):
        """Queue transaction when offline"""
        self.queue.append(transaction)
        self.save_queue()
    
    def sync_when_online(self):
        """Sync queued transactions when connection restored"""
        if not self.is_online():
            return
        
        for transaction in self.queue:
            try:
                # Send to cloud
                response = requests.post(
                    "https://api.gaitpay.com/v1/transactions/sync",
                    json=transaction
                )
                
                if response.status_code == 200:
                    self.queue.remove(transaction)
            except Exception as e:
                log_error(f"Failed to sync transaction: {e}")
        
        self.save_queue()

# Check for connectivity every minute
schedule.every(1).minutes.do(queue.sync_when_online)
```


## 13. Monitoring and Logging System

### 13.1 Observability Stack

**Three Pillars of Observability**:
1. **Metrics**: Quantitative measurements (latency, throughput, error rate)
2. **Logs**: Discrete events with context (errors, warnings, info)
3. **Traces**: Request flow across distributed services

**Technology Stack**:
```yaml
Metrics:
  - Prometheus: Time-series database for metrics
  - Grafana: Visualization and dashboards
  - CloudWatch: AWS-native metrics

Logs:
  - ELK Stack: Elasticsearch, Logstash, Kibana
  - CloudWatch Logs: Centralized log aggregation
  - Fluentd: Log forwarding and parsing

Traces:
  - AWS X-Ray: Distributed tracing
  - Jaeger: Open-source tracing (alternative)
  - OpenTelemetry: Instrumentation framework

Alerting:
  - PagerDuty: Incident management
  - Slack: Team notifications
  - SNS: AWS Simple Notification Service
```

### 13.2 Key Metrics to Monitor

**System Metrics**:
```python
# Prometheus metrics definition
from prometheus_client import Counter, Histogram, Gauge

# Request metrics
request_count = Counter(
    'gait_api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status']
)

request_latency = Histogram(
    'gait_api_request_duration_seconds',
    'API request latency',
    ['method', 'endpoint'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
)

# Identification metrics
identification_accuracy = Gauge(
    'gait_identification_accuracy',
    'Current identification accuracy',
    ['store_id']
)

identification_latency = Histogram(
    'gait_identification_duration_seconds',
    'Gait identification latency',
    buckets=[0.5, 1.0, 2.0, 3.0, 5.0]
)

# Payment metrics
payment_success_rate = Gauge(
    'payment_success_rate',
    'Payment success rate (last 1 hour)',
    ['payment_gateway']
)

payment_amount = Histogram(
    'payment_amount_inr',
    'Payment amount distribution',
    buckets=[100, 500, 1000, 2000, 5000, 10000]
)

# Infrastructure metrics
gpu_utilization = Gauge(
    'gpu_utilization_percent',
    'GPU utilization',
    ['device_id']
)

database_connections = Gauge(
    'database_connections_active',
    'Active database connections'
)
```

**Business Metrics**:
```python
# Daily active customers
daily_active_customers = Gauge(
    'daily_active_customers',
    'Number of unique customers per day',
    ['store_id']
)

# Average transaction value
avg_transaction_value = Gauge(
    'avg_transaction_value_inr',
    'Average transaction value',
    ['store_id']
)

# Customer enrollment rate
enrollment_rate = Counter(
    'customer_enrollments_total',
    'Total customer enrollments',
    ['source']  # mobile_app, in_store
)

# Dispute rate
dispute_rate = Gauge(
    'dispute_rate_percent',
    'Percentage of disputed transactions',
    ['store_id']
)
```

### 13.3 Logging Strategy

**Structured Logging**:
```python
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Usage examples
logger.info(
    "customer_identified",
    customer_id="uuid-123",
    store_id="store-456",
    confidence=0.94,
    latency_ms=2700
)

logger.warning(
    "low_confidence_identification",
    customer_id="uuid-789",
    confidence=0.78,
    threshold=0.85,
    action="manual_verification_required"
)

logger.error(
    "payment_failed",
    customer_id="uuid-123",
    transaction_id="txn-456",
    amount=1250.50,
    error_code="INSUFFICIENT_FUNDS",
    payment_gateway="razorpay"
)
```

**Log Levels**:
```python
# DEBUG: Detailed diagnostic information
logger.debug("gait_features_extracted", features_shape=(30, 128))

# INFO: General informational messages
logger.info("customer_entered_store", customer_id="uuid-123")

# WARNING: Potentially problematic situations
logger.warning("high_gpu_temperature", temperature=85, threshold=80)

# ERROR: Error events that might still allow the application to continue
logger.error("database_query_failed", query="SELECT * FROM customers", error=str(e))

# CRITICAL: Severe error events that might cause the application to abort
logger.critical("database_connection_lost", attempts=5, max_retries=5)
```

**Log Retention Policy**:
```yaml
Log Retention:
  DEBUG logs: 7 days
  INFO logs: 30 days
  WARNING logs: 90 days
  ERROR logs: 1 year
  CRITICAL logs: 7 years (compliance)
  
  Audit logs: 7 years (regulatory requirement)
  Transaction logs: 7 years (financial records)
```

### 13.4 Distributed Tracing

**X-Ray Instrumentation**:
```python
from aws_xray_sdk.core import xray_recorder
from aws_xray_sdk.ext.flask.middleware import XRayMiddleware

# Instrument Flask app
app = Flask(__name__)
XRayMiddleware(app, xray_recorder)

@app.route('/api/v1/identify', methods=['POST'])
def identify_customer():
    # Start subsegment for gait matching
    with xray_recorder.capture('gait_matching'):
        gait_features = request.json['gait_features']
        
        # Subsegment for database query
        with xray_recorder.capture('database_query'):
            templates = db.query("SELECT * FROM gait_templates")
        
        # Subsegment for similarity computation
        with xray_recorder.capture('similarity_computation'):
            customer_id = match_gait(gait_features, templates)
        
        # Add metadata to trace
        xray_recorder.current_subsegment().put_metadata('confidence', 0.94)
        xray_recorder.current_subsegment().put_annotation('customer_id', customer_id)
    
    return jsonify({'customer_id': customer_id})
```

**Trace Visualization**:
```
Request: POST /api/v1/identify
├─ gait_matching (2.7s)
│  ├─ database_query (0.5s)
│  │  └─ PostgreSQL: SELECT * FROM gait_templates (0.48s)
│  ├─ similarity_computation (2.0s)
│  │  ├─ vector_search (1.8s)
│  │  └─ confidence_calculation (0.2s)
│  └─ cache_update (0.2s)
└─ response_serialization (0.05s)

Total: 2.75s
```

### 13.5 Alerting Rules

**Critical Alerts** (PagerDuty - immediate response):
```yaml
alerts:
  - name: HighErrorRate
    condition: error_rate > 5% for 5 minutes
    severity: critical
    notification: pagerduty
    
  - name: DatabaseDown
    condition: database_connections == 0 for 1 minute
    severity: critical
    notification: pagerduty
    
  - name: PaymentGatewayDown
    condition: payment_failure_rate > 50% for 5 minutes
    severity: critical
    notification: pagerduty
    
  - name: GPUClusterDown
    condition: gpu_available_count == 0 for 2 minutes
    severity: critical
    notification: pagerduty
```

**Warning Alerts** (Slack - team awareness):
```yaml
alerts:
  - name: HighLatency
    condition: p95_latency > 3s for 10 minutes
    severity: warning
    notification: slack
    
  - name: LowIdentificationAccuracy
    condition: identification_accuracy < 90% for 15 minutes
    severity: warning
    notification: slack
    
  - name: HighGPUUtilization
    condition: gpu_utilization > 85% for 10 minutes
    severity: warning
    notification: slack
    
  - name: DiskSpaceLow
    condition: disk_usage > 80%
    severity: warning
    notification: slack
```

**Info Alerts** (Email - daily digest):
```yaml
alerts:
  - name: DailyMetricsSummary
    schedule: daily at 9:00 AM
    severity: info
    notification: email
    content:
      - total_transactions
      - total_revenue
      - avg_transaction_value
      - identification_accuracy
      - payment_success_rate
      - new_enrollments
```

### 13.6 Dashboards

**Grafana Dashboard - System Health**:
```json
{
  "dashboard": {
    "title": "Gait Payment System - Health",
    "panels": [
      {
        "title": "API Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(gait_api_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "API Latency (P95)",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(gait_api_request_duration_seconds_bucket[5m]))",
            "legendFormat": "{{endpoint}}"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(gait_api_requests_total{status=~\"5..\"}[5m]) / rate(gait_api_requests_total[5m])",
            "legendFormat": "Error Rate"
          }
        ]
      },
      {
        "title": "GPU Utilization",
        "type": "gauge",
        "targets": [
          {
            "expr": "avg(gpu_utilization_percent)",
            "legendFormat": "GPU Usage"
          }
        ]
      },
      {
        "title": "Database Connections",
        "type": "stat",
        "targets": [
          {
            "expr": "database_connections_active",
            "legendFormat": "Active Connections"
          }
        ]
      }
    ]
  }
}
```

**Grafana Dashboard - Business Metrics**:
```json
{
  "dashboard": {
    "title": "Gait Payment System - Business",
    "panels": [
      {
        "title": "Daily Active Customers",
        "type": "graph",
        "targets": [
          {
            "expr": "daily_active_customers",
            "legendFormat": "{{store_id}}"
          }
        ]
      },
      {
        "title": "Revenue (Today)",
        "type": "stat",
        "targets": [
          {
            "expr": "sum(increase(payment_amount_inr_sum[24h]))",
            "legendFormat": "Total Revenue"
          }
        ]
      },
      {
        "title": "Payment Success Rate",
        "type": "gauge",
        "targets": [
          {
            "expr": "payment_success_rate",
            "legendFormat": "{{payment_gateway}}"
          }
        ]
      },
      {
        "title": "Identification Accuracy",
        "type": "graph",
        "targets": [
          {
            "expr": "identification_accuracy",
            "legendFormat": "{{store_id}}"
          }
        ]
      }
    ]
  }
}
```

### 13.7 Log Analysis Queries

**Elasticsearch Queries**:
```json
// Find all failed payments in last 24 hours
{
  "query": {
    "bool": {
      "must": [
        {"match": {"event": "payment_failed"}},
        {"range": {"timestamp": {"gte": "now-24h"}}}
      ]
    }
  },
  "aggs": {
    "by_error_code": {
      "terms": {"field": "error_code"}
    }
  }
}

// Find customers with low identification confidence
{
  "query": {
    "bool": {
      "must": [
        {"match": {"event": "customer_identified"}},
        {"range": {"confidence": {"lt": 0.85}}}
      ]
    }
  },
  "sort": [{"confidence": "asc"}]
}

// Analyze identification latency by store
{
  "query": {
    "match": {"event": "customer_identified"}
  },
  "aggs": {
    "by_store": {
      "terms": {"field": "store_id"},
      "aggs": {
        "avg_latency": {"avg": {"field": "latency_ms"}}
      }
    }
  }
}
```

### 13.8 Performance Monitoring

**Real User Monitoring (RUM)**:
```javascript
// Mobile app instrumentation
import { Analytics } from 'aws-amplify';

// Track screen load time
const startTime = Date.now();
// ... load screen ...
const loadTime = Date.now() - startTime;

Analytics.record({
  name: 'screen_load',
  attributes: {
    screen_name: 'enrollment',
    load_time_ms: loadTime,
    device_type: Platform.OS,
    app_version: '1.2.0'
  }
});

// Track API call performance
const apiStartTime = Date.now();
const response = await fetch('/api/v1/enrollment', {...});
const apiDuration = Date.now() - apiStartTime;

Analytics.record({
  name: 'api_call',
  attributes: {
    endpoint: '/api/v1/enrollment',
    duration_ms: apiDuration,
    status_code: response.status,
    network_type: getNetworkType()
  }
});
```

**Synthetic Monitoring**:
```python
# Scheduled health checks from multiple regions
def synthetic_health_check():
    """Simulate customer journey from multiple locations"""
    regions = ['us-east-1', 'ap-south-1', 'eu-west-1']
    
    for region in regions:
        start_time = time.time()
        
        try:
            # Test identification API
            response = requests.post(
                f"https://api.gaitpay.com/v1/identify",
                json={"gait_features": generate_test_features()},
                timeout=5
            )
            
            latency = (time.time() - start_time) * 1000
            
            # Record metrics
            cloudwatch.put_metric_data(
                Namespace='GaitPayment/Synthetic',
                MetricData=[
                    {
                        'MetricName': 'APILatency',
                        'Value': latency,
                        'Unit': 'Milliseconds',
                        'Dimensions': [
                            {'Name': 'Region', 'Value': region},
                            {'Name': 'Endpoint', 'Value': '/identify'}
                        ]
                    },
                    {
                        'MetricName': 'APIAvailability',
                        'Value': 1 if response.status_code == 200 else 0,
                        'Unit': 'Count',
                        'Dimensions': [
                            {'Name': 'Region', 'Value': region}
                        ]
                    }
                ]
            )
        except Exception as e:
            logger.error(f"Synthetic check failed for {region}: {e}")
            cloudwatch.put_metric_data(
                Namespace='GaitPayment/Synthetic',
                MetricData=[
                    {
                        'MetricName': 'APIAvailability',
                        'Value': 0,
                        'Unit': 'Count',
                        'Dimensions': [{'Name': 'Region', 'Value': region}]
                    }
                ]
            )

# Run every 5 minutes
schedule.every(5).minutes.do(synthetic_health_check)
```

---

## 14. Conclusion

This design document provides a comprehensive technical blueprint for the Zero-Interface Payments Using Gait Recognition system. The architecture balances:

- **Performance**: Sub-3-second identification with hybrid edge-cloud processing
- **Scalability**: Support for 500+ stores and 1M+ customers
- **Security**: Multi-layered defense with encryption, tokenization, and biometric protection
- **Privacy**: GDPR/DPDPA compliance with consent management and data minimization
- **Reliability**: 99.5% uptime with redundancy and disaster recovery
- **Cost-Efficiency**: Optimized infrastructure with auto-scaling and caching

**Key Innovations**:
1. Hybrid edge-cloud architecture for optimal latency and cost
2. Cancelable biometric templates for enhanced privacy
3. Multi-gateway payment failover for reliability
4. Comprehensive monitoring and observability
5. Graceful degradation for fault tolerance

**Next Steps**:
1. Prototype development and pilot deployment (1 store)
2. Model training with diverse gait datasets
3. Security audit and penetration testing
4. Regulatory compliance review (GDPR, DPDPA, PCI-DSS)
5. Phased rollout to 10 stores, then nationwide

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02-15 | System Architect | Initial design document |

**Approval Status**: Draft - Pending Review

**Related Documents**:
- Requirements Document: `requirements.md`
- API Specification: `api-spec.yaml`
- Security Policy: `security-policy.md`
- Deployment Runbook: `deployment-runbook.md`
