export class API {
    static BASE_URL = '/api/v1';

    static async fetchJSON(endpoint, options = {}) {
        const response = await fetch(`${this.BASE_URL}${endpoint}`, {
            ...options,
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            }
        });
        if (!response.ok) throw new Error(await response.text());
        return response.json();
    }

    static labels = {
        list: (params = {}) => {
            const query = new URLSearchParams(params).toString();
            return this.fetchJSON(`/labels/?${query}`);
        },
        get: (filename) => this.fetchJSON(`/labels/${filename}`),
        save: (filename, data) => this.fetchJSON(`/labels/${filename}`, {
            method: 'POST',
            body: JSON.stringify(data)
        }),
        delete: (filename) => this.fetchJSON(`/labels/${filename}`, { method: 'DELETE' })
    };

    static versions = {
        list: () => this.fetchJSON('/versions/'),
        publish: (name, description) => this.fetchJSON('/versions/', {
            method: 'POST',
            body: JSON.stringify({ name, description })
        })
    };
}
