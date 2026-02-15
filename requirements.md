# Requirements Document: Zero-Interface Payments Using Gait Recognition

## 1. Problem Statement

Traditional retail payment systems require explicit user interaction through physical cards, mobile devices, QR codes, OTPs, or biometric scanners (fingerprint/face). These friction points create:

- **Queue bottlenecks** at checkout counters
- **Device dependency** requiring customers to carry phones or cards
- **Hygiene concerns** with shared touchpoints (fingerprint scanners, PIN pads)
- **Cognitive load** forcing customers to remember PINs or carry multiple payment instruments
- **Accessibility barriers** for elderly or differently-abled customers

Current "frictionless" solutions like Amazon Go still require app installation and explicit check-in. There is a need for a truly zero-interface payment system that identifies customers passively and processes transactions without any explicit action.

## 2. Objectives

### Primary Objectives
- Enable **automatic customer identification** using gait biometrics captured via overhead cameras
- Process **contactless payments** without requiring any device, card, or manual authentication
- Achieve **sub-3-second identification** latency from entry to recognition
- Maintain **>95% identification accuracy** in real-world retail environments
- Ensure **GDPR-compliant biometric data handling** with explicit consent mechanisms

### Secondary Objectives
- Reduce checkout time by 80% compared to traditional systems
- Support 100+ concurrent customers in a single retail location
- Enable seamless multi-store enrollment (enroll once, pay anywhere)
- Provide real-time fraud detection and anomaly alerts
- Create audit trails for dispute resolution

## 3. Scope

### 3.1 In-Scope

#### Customer Experience
- One-time enrollment via mobile app with consent capture
- Automatic identification upon store entry via gait recognition
- Real-time item tracking using computer vision (shelf sensors/cameras)
- Automatic payment deduction upon exit
- Receipt delivery via email/SMS/app notification
- Opt-out mechanism and data deletion requests

#### Technical Implementation
- Overhead camera network for gait capture (RGB + depth sensors)
- Pose estimation and gait feature extraction pipeline
- Deep learning model (CNN + LSTM) for gait classification
- Secure biometric template storage with encryption
- Integration with UPI/bank payment gateways
- Cloud-based processing with edge computing for latency-critical tasks
- Admin dashboard for store operators
- Monitoring and alerting infrastructure

#### Security & Compliance
- End-to-end encryption of biometric data
- Tokenized payment processing
- GDPR/DPDPA compliance framework
- Audit logging and data retention policies
- Fraud detection algorithms

### 3.2 Out-of-Scope

- **Facial recognition** (privacy concerns, regulatory restrictions)
- **Item-level RFID tagging** (cost prohibitive for small retailers)
- **Manual checkout fallback** (handled by existing store infrastructure)
- **Inventory management system** (integration only, not core feature)
- **Dynamic pricing engine** (separate business logic)
- **Customer behavior analytics** (future enhancement)
- **Multi-person household accounts** (Phase 2 feature)
- **Offline payment processing** (requires network connectivity)

## 4. Functional Requirements

### FR-1: Customer Enrollment
- **FR-1.1**: Mobile app shall capture gait signature via 30-second walking video
- **FR-1.2**: System shall extract 128-dimensional gait feature vector during enrollment
- **FR-1.3**: User shall provide explicit consent for biometric data collection (checkbox + signature)
- **FR-1.4**: System shall link gait template to payment method (UPI/card/bank account)
- **FR-1.5**: Enrollment shall require government ID verification (Aadhaar/PAN/Passport)
- **FR-1.6**: System shall support re-enrollment if gait pattern changes (injury, footwear change)

### FR-2: Store Entry & Identification
- **FR-2.1**: Overhead cameras shall capture customer gait within 2 meters of entry
- **FR-2.2**: System shall process 15 FPS video stream for pose estimation
- **FR-2.3**: Gait feature extraction shall complete within 1.5 seconds
- **FR-2.4**: AI model shall match gait signature against database within 1 second
- **FR-2.5**: System shall display silent confirmation (optional LED indicator at entry)
- **FR-2.6**: Unidentified customers shall trigger fallback notification to staff

### FR-3: Item Tracking
- **FR-3.1**: System shall integrate with existing shelf sensors/cameras for item detection
- **FR-3.2**: Each item pickup shall be associated with identified customer
- **FR-3.3**: System shall handle multiple customers in same aisle (spatial tracking)
- **FR-3.4**: Cart shall update in real-time in customer's mobile app (optional)

### FR-4: Payment Processing
- **FR-4.1**: System shall calculate total bill upon customer crossing exit threshold
- **FR-4.2**: Payment shall be deducted automatically via pre-linked payment method
- **FR-4.3**: Transaction shall complete within 2 seconds of exit
- **FR-4.4**: System shall send digital receipt via email/SMS/push notification
- **FR-4.5**: Failed payments shall trigger alert to store staff and block exit (optional gate)

### FR-5: User Management
- **FR-5.1**: Customers shall view transaction history via mobile app
- **FR-5.2**: Customers shall update payment methods without re-enrollment
- **FR-5.3**: Customers shall request data deletion (GDPR right to be forgotten)
- **FR-5.4**: System shall disable account after 3 consecutive failed payments
- **FR-5.5**: Customers shall set spending limits and receive alerts

### FR-6: Admin Dashboard
- **FR-6.1**: Store operators shall view real-time customer count and identification status
- **FR-6.2**: System shall provide daily transaction reports and reconciliation
- **FR-6.3**: Admins shall manually verify disputed transactions via video playback
- **FR-6.4**: System shall flag anomalies (unidentified persons, loitering, unusual patterns)

### FR-7: Fraud Detection
- **FR-7.1**: System shall detect gait spoofing attempts (video replay, prosthetics)
- **FR-7.2**: Liveness detection shall verify real-time human presence
- **FR-7.3**: System shall flag accounts with abnormal transaction patterns
- **FR-7.4**: Multiple failed identification attempts shall trigger security alert

## 5. Non-Functional Requirements

### NFR-1: Performance
- **NFR-1.1**: Identification latency: <3 seconds (entry to recognition)
- **NFR-1.2**: Payment processing: <2 seconds (exit to deduction)
- **NFR-1.3**: System throughput: 100+ concurrent customers per store
- **NFR-1.4**: Camera frame processing: 15 FPS minimum
- **NFR-1.5**: API response time: <500ms (95th percentile)

### NFR-2: Accuracy
- **NFR-2.1**: Gait recognition accuracy: >95% (True Positive Rate)
- **NFR-2.2**: False Positive Rate: <2% (misidentification)
- **NFR-2.3**: False Negative Rate: <5% (failed identification)
- **NFR-2.4**: Item tracking accuracy: >98%

### NFR-3: Scalability
- **NFR-3.1**: System shall support 10,000+ enrolled customers per store
- **NFR-3.2**: Database shall handle 1M+ gait templates across all stores
- **NFR-3.3**: Infrastructure shall auto-scale during peak hours (evenings, weekends)
- **NFR-3.4**: System shall support 100+ store locations on single cloud deployment

### NFR-4: Availability
- **NFR-4.1**: System uptime: 99.5% (excluding planned maintenance)
- **NFR-4.2**: Graceful degradation: fallback to manual checkout if system fails
- **NFR-4.3**: Database replication: multi-region with <1-minute failover

### NFR-5: Security
- **NFR-5.1**: Biometric templates encrypted at rest (AES-256)
- **NFR-5.2**: Data in transit encrypted via TLS 1.3
- **NFR-5.3**: Payment tokenization (no raw card/account numbers stored)
- **NFR-5.4**: Role-based access control (RBAC) for admin dashboard
- **NFR-5.5**: Penetration testing quarterly, vulnerability patching within 48 hours

### NFR-6: Privacy & Compliance
- **NFR-6.1**: GDPR Article 9 compliance (biometric data as special category)
- **NFR-6.2**: India DPDPA 2023 compliance (consent, data localization)
- **NFR-6.3**: PCI-DSS compliance for payment processing
- **NFR-6.4**: Data retention: biometric templates deleted within 30 days of account closure
- **NFR-6.5**: Audit logs retained for 7 years (regulatory requirement)

### NFR-7: Usability
- **NFR-7.1**: Enrollment process: <5 minutes end-to-end
- **NFR-7.2**: Mobile app: support iOS 14+ and Android 10+
- **NFR-7.3**: Admin dashboard: accessible via web browser (Chrome, Firefox, Safari)
- **NFR-7.4**: Multi-language support: English, Hindi, regional languages

### NFR-8: Maintainability
- **NFR-8.1**: Modular architecture with microservices
- **NFR-8.2**: CI/CD pipeline for automated testing and deployment
- **NFR-8.3**: Comprehensive logging (ELK stack or equivalent)
- **NFR-8.4**: Model retraining pipeline with A/B testing framework

## 6. User Roles

### 6.1 Customer
- Enroll via mobile app with consent
- Shop in enabled stores without explicit checkout
- View transaction history and receipts
- Manage payment methods and preferences
- Request data deletion

### 6.2 Store Operator
- Monitor real-time customer identification status
- Handle failed payment alerts
- Resolve disputes via video playback
- Generate daily reconciliation reports

### 6.3 System Administrator
- Manage store configurations and camera networks
- Monitor system health and performance metrics
- Deploy model updates and system patches
- Configure fraud detection rules

### 6.4 Data Protection Officer (DPO)
- Handle GDPR/DPDPA compliance requests
- Audit data access logs
- Manage consent records
- Oversee data deletion workflows

### 6.5 ML Engineer
- Train and evaluate gait recognition models
- Monitor model drift and accuracy degradation
- Implement A/B tests for model improvements
- Optimize inference latency

## 7. System Constraints

### 7.1 Hardware Constraints
- Minimum 4 overhead cameras per 500 sq ft store area
- Camera specifications: 1080p resolution, 30 FPS, depth sensor (optional)
- Network bandwidth: 10 Mbps upload per camera
- Edge device: NVIDIA Jetson or equivalent (optional for local processing)

### 7.2 Environmental Constraints
- Adequate lighting: 300+ lux (standard retail lighting)
- Camera mounting height: 8-12 feet for optimal gait capture
- Unobstructed walking path: 2-meter clear zone at entry/exit

### 7.3 Regulatory Constraints
- Explicit consent required before biometric capture
- Signage mandated at store entry (biometric surveillance notice)
- Data localization: Indian customer data stored in India (DPDPA requirement)
- Minor customers (<18 years): parental consent required

### 7.4 Technical Constraints
- Internet connectivity required (no offline mode)
- Payment gateway dependency (UPI/bank API availability)
- Model inference: GPU-accelerated servers for real-time processing
- Database: PostgreSQL or equivalent with vector search extension

## 8. Security Requirements

### 8.1 Authentication & Authorization
- Multi-factor authentication (MFA) for admin dashboard
- OAuth 2.0 for mobile app authentication
- API key rotation every 90 days
- Role-based access control with principle of least privilege

### 8.2 Data Protection
- Biometric templates stored as irreversible hashes (template protection)
- Payment credentials tokenized (never stored in raw form)
- Personal data encrypted with customer-specific keys
- Secure key management via HSM or cloud KMS

### 8.3 Network Security
- API gateway with rate limiting (100 requests/minute per user)
- DDoS protection via CDN (Cloudflare/AWS Shield)
- Intrusion detection system (IDS) monitoring
- VPN for admin access to production systems

### 8.4 Application Security
- Input validation and sanitization (prevent SQL injection, XSS)
- Secure coding practices (OWASP Top 10 compliance)
- Dependency scanning for vulnerable libraries
- Regular security audits and penetration testing

### 8.5 Incident Response
- Security incident response plan (SIRP) documented
- Breach notification within 72 hours (GDPR requirement)
- Forensic logging enabled for security events
- Disaster recovery plan with RTO <4 hours, RPO <1 hour

## 9. Privacy & Compliance Considerations

### 9.1 GDPR Compliance (EU Customers)
- **Lawful basis**: Explicit consent (Article 6) + special category consent (Article 9)
- **Data minimization**: Only gait features stored, not raw video
- **Purpose limitation**: Biometric data used solely for payment authentication
- **Right to access**: Customers can download their data via app
- **Right to erasure**: Account deletion removes all biometric templates within 30 days
- **Data portability**: Export transaction history in machine-readable format
- **Privacy by design**: Encryption and pseudonymization by default

### 9.2 India DPDPA 2023 Compliance
- **Consent management**: Clear, specific, informed consent with withdrawal option
- **Data localization**: Indian customer data stored in Indian data centers
- **Data fiduciary obligations**: Transparent privacy policy, grievance redressal
- **Children's data**: Parental consent for users under 18
- **Data breach notification**: Notify Data Protection Board within 72 hours

### 9.3 PCI-DSS Compliance (Payment Data)
- Tokenization of payment credentials (no card numbers stored)
- Secure transmission of payment data (TLS 1.3)
- Regular security assessments (quarterly scans, annual audits)
- Access control and monitoring

### 9.4 Biometric Data Handling
- **Consent capture**: Video recording of consent during enrollment
- **Template protection**: Biometric templates stored as cancelable hashes
- **No raw video storage**: Only pose keypoints and gait features retained
- **Access logging**: All biometric data access logged with user ID and timestamp
- **Third-party restrictions**: No sharing of biometric data with external parties

### 9.5 Transparency & Signage
- Clear signage at store entry: "Biometric surveillance in operation"
- Privacy policy accessible via QR code at entry
- Opt-out mechanism: customers can use traditional checkout
- Regular privacy impact assessments (PIAs)

## 10. Performance Requirements

### 10.1 Latency Requirements
| Operation | Target Latency | Maximum Acceptable |
|-----------|----------------|-------------------|
| Gait capture (entry) | 1.5s | 3s |
| Feature extraction | 0.8s | 1.5s |
| Database lookup | 0.5s | 1s |
| Payment processing | 1.5s | 3s |
| End-to-end (entry to ID) | 2.5s | 5s |

### 10.2 Throughput Requirements
- **Peak hour capacity**: 200 customers/hour per store
- **Concurrent identifications**: 10 simultaneous entry/exit events
- **Database queries**: 1000 QPS (queries per second)
- **API throughput**: 5000 requests/minute

### 10.3 Accuracy Requirements
| Metric | Target | Minimum Acceptable |
|--------|--------|-------------------|
| True Positive Rate (TPR) | 97% | 95% |
| False Positive Rate (FPR) | 1% | 2% |
| False Negative Rate (FNR) | 3% | 5% |
| Item tracking accuracy | 99% | 98% |

### 10.4 Resource Utilization
- **GPU utilization**: <70% during peak hours (headroom for spikes)
- **CPU utilization**: <60% average
- **Memory usage**: <80% of allocated RAM
- **Network bandwidth**: <50% of available capacity

### 10.5 Model Performance
- **Inference time**: <100ms per gait sequence (single customer)
- **Model size**: <500MB (for edge deployment)
- **Training time**: <24 hours for full model retraining
- **Accuracy degradation**: <2% per year (model drift monitoring)

## 11. Assumptions

### 11.1 Customer Behavior
- Customers walk naturally at entry/exit (not running or crawling)
- Customers wear similar footwear as during enrollment (or re-enroll)
- Customers do not intentionally obscure gait (e.g., exaggerated limping)
- Customers carry smartphones for enrollment and receipt delivery

### 11.2 Store Environment
- Retail stores have adequate lighting (300+ lux)
- Entry/exit points are clearly defined with camera coverage
- Store layout allows 2-meter unobstructed walking path
- Existing item tracking infrastructure (shelf sensors/cameras) is available

### 11.3 Technical Infrastructure
- Reliable internet connectivity (99% uptime)
- Cloud infrastructure availability (AWS/Azure/GCP)
- Payment gateway APIs operational (UPI/bank APIs)
- Sufficient power supply for camera network and edge devices

### 11.4 Regulatory Environment
- Biometric payment systems are legally permissible in target markets
- Consent-based biometric collection is compliant with local laws
- Payment regulations allow automated deductions with pre-authorization

### 11.5 Business Model
- Customers are willing to enroll for convenience benefits
- Retailers are willing to invest in camera infrastructure
- Transaction fees are acceptable to both customers and retailers
- Insurance coverage available for fraud/dispute liabilities

## 12. Future Enhancements

### Phase 2 (6-12 months)
- **Multi-person household accounts**: Family members share payment method
- **Gait + face fusion**: Combine modalities for higher accuracy (where legally permitted)
- **Behavioral analytics**: Heatmaps, dwell time, product affinity (with consent)
- **Dynamic pricing**: Personalized offers based on purchase history
- **Voice assistant integration**: "Alexa, what's my cart total?"

### Phase 3 (12-24 months)
- **Cross-retailer network**: Enroll once, pay at any partner store
- **Wearable integration**: Smartwatch notifications for cart updates
- **Augmented reality**: AR app showing virtual cart and product info
- **Predictive restocking**: AI-driven inventory management based on customer patterns
- **Gamification**: Loyalty points, challenges, social sharing

### Phase 4 (24+ months)
- **Autonomous stores**: Fully unmanned retail with robotic restocking
- **Drone delivery integration**: Purchase in-store, deliver to home
- **Blockchain receipts**: Immutable transaction records for warranty/returns
- **Emotion recognition**: Detect customer satisfaction (ethical considerations)
- **Gait health monitoring**: Alert customers to gait abnormalities (medical partnership)

### Research & Innovation
- **Federated learning**: Train models without centralizing biometric data
- **Homomorphic encryption**: Perform matching on encrypted templates
- **Quantum-resistant cryptography**: Future-proof security
- **Synthetic gait generation**: Augment training data while preserving privacy
- **Explainable AI**: Interpretable gait recognition for dispute resolution

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02-15 | System Architect | Initial requirements document |

**Approval Status**: Draft - Pending Review

**Next Steps**:
1. Stakeholder review and feedback incorporation
2. Technical feasibility assessment
3. Cost-benefit analysis
4. Regulatory compliance audit
5. Proceed to design phase
