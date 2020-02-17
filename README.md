# sgdnlu

[![Join the chat at https://gitter.im/schema-guided-dialog/community](https://badges.gitter.im/schema-guided-dialog/community.svg)](https://gitter.im/schema-guided-dialog/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

This is an open source nlu implmentation for [schema guided dialog](https://drive.google.com/file/d/1rn9g-1uWN6v4u6mwjf4xwEqKKhsfBSdW/view) settings. The implementation is based on the wonderful workshop on DSTC 8 that held during AAAI20. We focus on simpliciy and inference speed first, but also pay attention to performance, so the solution is more useful for practical applications. 

The main assumption of schema guided dialog is that chatbot is just conversational user interface to services, and conversational interface is about building common understanding of what user exact want so that system can serve them well. So the goal for conversational interface is filling frames (commonly known as intent and its slots that together captured what user wants) or collectively filling slots, and dialog is exchange of direct or indirect operations, represented in natural language, that manipulate these frames (or structured dialog states).  

This implementation is primarily influenced by two workshop paper:
1. [A BERT-based Unified Span Detection Framework for Schema-Guided Dialogue State Tracking](https://drive.google.com/file/d/1PRPx3lfJTtX-V23uTFOIPdB0LhQDBFq5/view), for it simplicity.
2. [The SPPD System for Schema Guided Dialogue State Tracking Challenge](https://drive.google.com/file/d/1wn8OTYrqyVVhwq-6XYWuhGN3t8mJohJr/view), for its through analysis of the dataset.

