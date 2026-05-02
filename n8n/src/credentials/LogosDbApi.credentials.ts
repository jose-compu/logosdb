import type { ICredentialType, INodeProperties } from 'n8n-workflow';

export class LogosDbApi implements ICredentialType {
  name = 'logosDbApi';
  displayName = 'LogosDB API';
  documentationUrl = 'https://github.com/jose-compu/logosdb';

  properties: INodeProperties[] = [
    {
      displayName: 'Database Path',
      name: 'dbPath',
      type: 'string',
      default: './.logosdb',
      description: 'Root directory where LogosDB stores its files',
    },
    {
      displayName: 'Embedding Provider',
      name: 'embeddingProvider',
      type: 'options',
      options: [
        { name: 'OpenAI', value: 'openai' },
        { name: 'Voyage AI', value: 'voyage' },
        { name: 'None (provide vectors manually)', value: 'none' },
      ],
      default: 'openai',
    },
    {
      displayName: 'OpenAI API Key',
      name: 'openAiApiKey',
      type: 'string',
      typeOptions: { password: true },
      default: '',
      displayOptions: {
        show: { embeddingProvider: ['openai'] },
      },
    },
    {
      displayName: 'Voyage API Key',
      name: 'voyageApiKey',
      type: 'string',
      typeOptions: { password: true },
      default: '',
      displayOptions: {
        show: { embeddingProvider: ['voyage'] },
      },
    },
  ];
}
