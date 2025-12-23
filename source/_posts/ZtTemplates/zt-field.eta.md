title: "<%= it.title %>"
created: <%= it.dateAdded %> 
modified: <%= it.dateModified %>
tags: <% it.tags.forEach(($it, i) => { %> <%= $it %> <% }) %>
collections: <% for (const $it of it.collections) { %> <%= $it %> <% } %>
year: <%= it.date %>
publication: <%= it.publicationTitle %>
citekey: "<%= it.citekey %>"
author: [<%= it.authors %>]